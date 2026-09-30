/*
 * k3_rdma.c - raw ibverbs Reliable-Connected (RC) RoCEv2 side-channel.
 *
 * A tiny single-endpoint transport for a 2-host request/response channel
 * between an aarch64 GB10 (DGX Spark) and an x86_64 RTX 3090 box.  It
 * mirrors the QP state machine and GID selection of the ds41rt native
 * reference (native/src/ds41rt_native.cc).
 *
 * Design notes:
 *   - RoCEv2 only.  GID auto-selection prefers IBV_GID_TYPE_ROCE_V2 with
 *     an IPv4-mapped address, falling back exactly like select_rc_gid().
 *   - One global endpoint per process (not thread safe for open/close).
 *   - Two CQs (one send, one recv) so wait_send()/wait_recv() never
 *     consume each other's completions.
 *   - All functions return 0 on success, -1 on failure, except
 *     k3rdma_wait_recv() which returns the received byte length and
 *     k3rdma_poll_recv() which returns 1 on a completed receive (storing
 *     the length in *out_len), 0 when the recv CQ is empty, and -1 on
 *     failure.
 *   - Errors are reported through k3rdma_last_error() (thread-local).
 *
 * Build: gcc -O2 -fPIC -shared -o libk3rdma.so k3_rdma.c -libverbs
 */

#define _GNU_SOURCE
#include <infiniband/verbs.h>

#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* Error reporting                                                     */
/* ------------------------------------------------------------------ */

/* Defined at the bottom; declared early because k3rdma_open() may call it. */
void k3rdma_close(void);
const char* k3rdma_last_error(void);

static __thread char g_last_error[512] = {0};

static void k3_set_error(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(g_last_error, sizeof(g_last_error), fmt, ap);
  va_end(ap);
}

const char* k3rdma_last_error(void) { return g_last_error; }

/* ------------------------------------------------------------------ */
/* Global endpoint                                                     */
/* ------------------------------------------------------------------ */

#define K3_SEND_WR_ID 0x5E4D0001u
#define K3_RECV_WR_ID 0x5E520001u

struct k3_endpoint {
  int open;

  struct ibv_device** dev_list;
  struct ibv_context* context;
  struct ibv_pd* pd;
  struct ibv_cq* send_cq;
  struct ibv_cq* recv_cq;
  struct ibv_qp* qp;

  struct ibv_port_attr port_attr;
  uint32_t port_num;
  uint32_t psn;
  uint32_t lid;
  uint32_t gid_index;
  union ibv_gid gid;
};

static struct k3_endpoint g_ep;

/* ------------------------------------------------------------------ */
/* GID helpers (mirror ds41rt select_rc_gid)                           */
/* ------------------------------------------------------------------ */

static int gid_is_zero(const union ibv_gid* gid) {
  for (int i = 0; i < 16; ++i) {
    if (gid->raw[i] != 0) {
      return 0;
    }
  }
  return 1;
}

static int gid_is_ipv4_mapped(const union ibv_gid* gid) {
  for (int i = 0; i < 10; ++i) {
    if (gid->raw[i] != 0) {
      return 0;
    }
  }
  return gid->raw[10] == 0xff && gid->raw[11] == 0xff;
}

/*
 * Select a usable RoCE GID.  Prefer RoCEv2 + IPv4-mapped; otherwise use
 * the ds41rt fallback ordering.  Returns 0 on success.
 */
static int k3_select_rc_gid(struct ibv_context* context,
                            const struct ibv_port_attr* port_attr,
                            uint32_t port_num, union ibv_gid* out_gid,
                            uint32_t* out_gid_index) {
  if (context == NULL || port_attr == NULL || out_gid == NULL ||
      out_gid_index == NULL) {
    k3_set_error("GID selection: null argument");
    return -1;
  }

  if (port_attr->link_layer != IBV_LINK_LAYER_ETHERNET) {
    if (ibv_query_gid(context, (uint8_t)port_num, 0, out_gid) != 0) {
      k3_set_error("ibv_query_gid failed for non-Ethernet port");
      return -1;
    }
    *out_gid_index = 0;
    return 0;
  }

  int have_fallback = 0;
  union ibv_gid fallback_gid;
  uint32_t fallback_index = 0;
  memset(&fallback_gid, 0, sizeof(fallback_gid));

  uint32_t gid_count = port_attr->gid_tbl_len;
  if (gid_count < 1) {
    gid_count = 1;
  }

  for (uint32_t index = 0; index < gid_count; ++index) {
    struct ibv_gid_entry entry;
    memset(&entry, 0, sizeof(entry));
    if (ibv_query_gid_ex(context, port_num, index, &entry, 0) != 0) {
      continue;
    }
    if (gid_is_zero(&entry.gid)) {
      continue;
    }
    const int ipv4_mapped = gid_is_ipv4_mapped(&entry.gid);
    if (entry.gid_type == IBV_GID_TYPE_ROCE_V2 && ipv4_mapped) {
      *out_gid = entry.gid;
      *out_gid_index = index;
      return 0;
    }
    if (!have_fallback ||
        (entry.gid_type == IBV_GID_TYPE_ROCE_V2 &&
         (ipv4_mapped || !gid_is_ipv4_mapped(&fallback_gid))) ||
        (ipv4_mapped && !gid_is_ipv4_mapped(&fallback_gid))) {
      fallback_gid = entry.gid;
      fallback_index = index;
      have_fallback = 1;
    }
  }

  if (!have_fallback) {
    k3_set_error("no usable RoCE GID found");
    return -1;
  }
  *out_gid = fallback_gid;
  *out_gid_index = fallback_index;
  return 0;
}

/* ------------------------------------------------------------------ */
/* QP state machine (mirror ds41rt)                                    */
/* ------------------------------------------------------------------ */

static int k3_modify_qp_to_init(struct ibv_qp* qp, uint32_t port_num) {
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = (uint8_t)port_num;
  attr.qp_access_flags =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
  const int flags =
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
  if (ibv_modify_qp(qp, &attr, flags) != 0) {
    k3_set_error("ibv_modify_qp INIT failed: %s", strerror(errno));
    return -1;
  }
  return 0;
}

static int k3_modify_qp_to_rtr(struct ibv_qp* qp,
                               const struct ibv_port_attr* port_attr,
                               uint32_t port_num, uint32_t remote_qp_num,
                               uint32_t remote_psn, uint32_t remote_lid,
                               const union ibv_gid* remote_gid,
                               uint32_t local_gid_index) {
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = port_attr->active_mtu;
  attr.dest_qp_num = remote_qp_num;
  attr.rq_psn = remote_psn;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;
  attr.ah_attr.dlid =
      remote_lid != 0 ? (uint16_t)remote_lid : port_attr->lid;
  attr.ah_attr.sl = 0;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num = (uint8_t)port_num;
  if (port_attr->link_layer == IBV_LINK_LAYER_ETHERNET) {
    if (remote_gid == NULL) {
      k3_set_error("RTR: remote GID required for Ethernet link layer");
      return -1;
    }
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid = *remote_gid;
    attr.ah_attr.grh.sgid_index = local_gid_index;
    attr.ah_attr.grh.hop_limit = 1;
  }
  const int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                    IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                    IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
  if (ibv_modify_qp(qp, &attr, flags) != 0) {
    k3_set_error("ibv_modify_qp RTR failed: %s", strerror(errno));
    return -1;
  }
  return 0;
}

static int k3_modify_qp_to_rts(struct ibv_qp* qp, uint32_t local_psn) {
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.sq_psn = local_psn;
  attr.max_rd_atomic = 1;
  const int flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                    IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
  if (ibv_modify_qp(qp, &attr, flags) != 0) {
    k3_set_error("ibv_modify_qp RTS failed: %s", strerror(errno));
    return -1;
  }
  return 0;
}

/* ------------------------------------------------------------------ */
/* Info string formatting / parsing                                    */
/* ------------------------------------------------------------------ */

static void k3_format_gid(const union ibv_gid* gid, char* out, size_t cap) {
  snprintf(out, cap,
           "%04x:%04x:%04x:%04x:%04x:%04x:%04x:%04x",
           (unsigned)((gid->raw[0] << 8) | gid->raw[1]),
           (unsigned)((gid->raw[2] << 8) | gid->raw[3]),
           (unsigned)((gid->raw[4] << 8) | gid->raw[5]),
           (unsigned)((gid->raw[6] << 8) | gid->raw[7]),
           (unsigned)((gid->raw[8] << 8) | gid->raw[9]),
           (unsigned)((gid->raw[10] << 8) | gid->raw[11]),
           (unsigned)((gid->raw[12] << 8) | gid->raw[13]),
           (unsigned)((gid->raw[14] << 8) | gid->raw[15]));
}

static int k3_parse_gid(const char* text, union ibv_gid* out_gid) {
  unsigned w[8];
  if (sscanf(text, "%x:%x:%x:%x:%x:%x:%x:%x", &w[0], &w[1], &w[2], &w[3],
             &w[4], &w[5], &w[6], &w[7]) != 8) {
    return -1;
  }
  for (int i = 0; i < 8; ++i) {
    out_gid->raw[2 * i] = (uint8_t)((w[i] >> 8) & 0xff);
    out_gid->raw[2 * i + 1] = (uint8_t)(w[i] & 0xff);
  }
  return 0;
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

int k3rdma_open(int is_server, const char* hca_name, int port_num,
                int gid_index, char* info_out, int info_out_cap) {
  if (g_ep.open) {
    k3_set_error("k3rdma_open: endpoint already open");
    return -1;
  }
  if (info_out == NULL || info_out_cap <= 0) {
    k3_set_error("k3rdma_open: invalid info_out buffer");
    return -1;
  }
  if (port_num <= 0 || port_num > 255) {
    k3_set_error("k3rdma_open: invalid port_num %d", port_num);
    return -1;
  }
  memset(&g_ep, 0, sizeof(g_ep));
  g_ep.port_num = (uint32_t)port_num;

  int rc = -1;
  struct ibv_device* device = NULL;

  g_ep.dev_list = ibv_get_device_list(NULL);
  if (g_ep.dev_list == NULL) {
    k3_set_error("ibv_get_device_list failed: %s", strerror(errno));
    goto cleanup;
  }

  if (hca_name != NULL && hca_name[0] != '\0') {
    for (int i = 0; g_ep.dev_list[i] != NULL; ++i) {
      if (strcmp(ibv_get_device_name(g_ep.dev_list[i]), hca_name) == 0) {
        device = g_ep.dev_list[i];
        break;
      }
    }
    if (device == NULL) {
      k3_set_error("HCA '%s' not found", hca_name);
      goto cleanup;
    }
  } else {
    if (g_ep.dev_list[0] == NULL) {
      k3_set_error("no RDMA devices present");
      goto cleanup;
    }
    device = g_ep.dev_list[0];
  }

  g_ep.context = ibv_open_device(device);
  if (g_ep.context == NULL) {
    k3_set_error("ibv_open_device failed: %s", strerror(errno));
    goto cleanup;
  }

  if (ibv_query_port(g_ep.context, (uint8_t)g_ep.port_num, &g_ep.port_attr) !=
      0) {
    k3_set_error("ibv_query_port failed: %s", strerror(errno));
    goto cleanup;
  }

  if (gid_index >= 0) {
    if (ibv_query_gid(g_ep.context, (uint8_t)g_ep.port_num,
                      (uint32_t)gid_index, &g_ep.gid) != 0) {
      k3_set_error("ibv_query_gid index %d failed: %s", gid_index,
                   strerror(errno));
      goto cleanup;
    }
    g_ep.gid_index = (uint32_t)gid_index;
  } else {
    if (k3_select_rc_gid(g_ep.context, &g_ep.port_attr, g_ep.port_num,
                         &g_ep.gid, &g_ep.gid_index) != 0) {
      goto cleanup;
    }
  }

  g_ep.pd = ibv_alloc_pd(g_ep.context);
  if (g_ep.pd == NULL) {
    k3_set_error("ibv_alloc_pd failed: %s", strerror(errno));
    goto cleanup;
  }

  g_ep.send_cq = ibv_create_cq(g_ep.context, 16, NULL, NULL, 0);
  if (g_ep.send_cq == NULL) {
    k3_set_error("ibv_create_cq (send) failed: %s", strerror(errno));
    goto cleanup;
  }
  g_ep.recv_cq = ibv_create_cq(g_ep.context, 16, NULL, NULL, 0);
  if (g_ep.recv_cq == NULL) {
    k3_set_error("ibv_create_cq (recv) failed: %s", strerror(errno));
    goto cleanup;
  }

  struct ibv_qp_init_attr qp_attr;
  memset(&qp_attr, 0, sizeof(qp_attr));
  qp_attr.send_cq = g_ep.send_cq;
  qp_attr.recv_cq = g_ep.recv_cq;
  qp_attr.qp_type = IBV_QPT_RC;
  qp_attr.cap.max_send_wr = 16;
  qp_attr.cap.max_recv_wr = 16;
  qp_attr.cap.max_send_sge = 1;
  qp_attr.cap.max_recv_sge = 1;
  g_ep.qp = ibv_create_qp(g_ep.pd, &qp_attr);
  if (g_ep.qp == NULL) {
    k3_set_error("ibv_create_qp failed: %s", strerror(errno));
    goto cleanup;
  }

  if (k3_modify_qp_to_init(g_ep.qp, g_ep.port_num) != 0) {
    goto cleanup;
  }

  g_ep.psn = is_server ? 0x414141u : 0x313131u;
  g_ep.lid = g_ep.port_attr.lid;

  char gid_text[64];
  k3_format_gid(&g_ep.gid, gid_text, sizeof(gid_text));
  int written = snprintf(info_out, (size_t)info_out_cap,
                         "qpn=%u;psn=0x%x;lid=%u;gid=%s;gid_index=%u",
                         g_ep.qp->qp_num, g_ep.psn, g_ep.lid, gid_text,
                         g_ep.gid_index);
  if (written < 0 || written >= info_out_cap) {
    k3_set_error("info_out buffer too small (%d bytes needed)", written + 1);
    goto cleanup;
  }

  g_ep.open = 1;
  return 0;

cleanup:
  k3rdma_close();
  return rc;
}

int k3rdma_connect(const char* peer_info) {
  if (!g_ep.open || g_ep.qp == NULL) {
    k3_set_error("k3rdma_connect: endpoint not open");
    return -1;
  }
  if (peer_info == NULL) {
    k3_set_error("k3rdma_connect: null peer_info");
    return -1;
  }

  unsigned qpn = 0, psn = 0, lid = 0, gid_index = 0;
  char gid_text[64];
  gid_text[0] = '\0';

  if (sscanf(peer_info,
             "qpn=%u;psn=0x%x;lid=%u;gid=%63[^;];gid_index=%u",
             &qpn, &psn, &lid, gid_text, &gid_index) != 5) {
    k3_set_error("k3rdma_connect: malformed peer_info '%s'", peer_info);
    return -1;
  }

  union ibv_gid remote_gid;
  memset(&remote_gid, 0, sizeof(remote_gid));
  if (k3_parse_gid(gid_text, &remote_gid) != 0) {
    k3_set_error("k3rdma_connect: malformed peer GID '%s'", gid_text);
    return -1;
  }

  if (k3_modify_qp_to_rtr(g_ep.qp, &g_ep.port_attr, g_ep.port_num, qpn, psn,
                          lid, &remote_gid, g_ep.gid_index) != 0) {
    return -1;
  }
  if (k3_modify_qp_to_rts(g_ep.qp, g_ep.psn) != 0) {
    return -1;
  }
  return 0;
}

void* k3rdma_register(void* buf, int len) {
  if (!g_ep.open || g_ep.pd == NULL) {
    k3_set_error("k3rdma_register: endpoint not open");
    return NULL;
  }
  if (buf == NULL || len <= 0) {
    k3_set_error("k3rdma_register: invalid buffer/len");
    return NULL;
  }
  struct ibv_mr* mr = ibv_reg_mr(g_ep.pd, buf, (size_t)len,
                                 IBV_ACCESS_LOCAL_WRITE |
                                     IBV_ACCESS_REMOTE_WRITE |
                                     IBV_ACCESS_REMOTE_READ);
  if (mr == NULL) {
    k3_set_error("ibv_reg_mr failed: %s", strerror(errno));
    return NULL;
  }
  return mr;
}

void k3rdma_deregister(void* mr) {
  if (mr != NULL) {
    ibv_dereg_mr((struct ibv_mr*)mr);
  }
}

int k3rdma_send(void* mr, int len) {
  if (!g_ep.open || g_ep.qp == NULL) {
    k3_set_error("k3rdma_send: endpoint not open");
    return -1;
  }
  if (mr == NULL || len <= 0) {
    k3_set_error("k3rdma_send: invalid MR/len");
    return -1;
  }
  struct ibv_mr* m = (struct ibv_mr*)mr;
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = (uintptr_t)m->addr;
  sge.length = (uint32_t)len;
  sge.lkey = m->lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = K3_SEND_WR_ID;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;

  struct ibv_send_wr* bad_wr = NULL;
  if (ibv_post_send(g_ep.qp, &wr, &bad_wr) != 0) {
    k3_set_error("ibv_post_send failed: %s", strerror(errno));
    return -1;
  }
  return 0;
}

int k3rdma_post_recv(void* mr, int maxlen) {
  if (!g_ep.open || g_ep.qp == NULL) {
    k3_set_error("k3rdma_post_recv: endpoint not open");
    return -1;
  }
  if (mr == NULL || maxlen <= 0) {
    k3_set_error("k3rdma_post_recv: invalid MR/maxlen");
    return -1;
  }
  struct ibv_mr* m = (struct ibv_mr*)mr;
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = (uintptr_t)m->addr;
  sge.length = (uint32_t)maxlen;
  sge.lkey = m->lkey;

  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = K3_RECV_WR_ID;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  struct ibv_recv_wr* bad_wr = NULL;
  if (ibv_post_recv(g_ep.qp, &wr, &bad_wr) != 0) {
    k3_set_error("ibv_post_recv failed: %s", strerror(errno));
    return -1;
  }
  return 0;
}

static int64_t k3_now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

int k3rdma_wait_send(int timeout_ms) {
  if (!g_ep.open || g_ep.send_cq == NULL) {
    k3_set_error("k3rdma_wait_send: endpoint not open");
    return -1;
  }
  const int64_t deadline = k3_now_ms() + (timeout_ms > 0 ? timeout_ms : 0);
  for (;;) {
    struct ibv_wc wc;
    memset(&wc, 0, sizeof(wc));
    int polled = ibv_poll_cq(g_ep.send_cq, 1, &wc);
    if (polled < 0) {
      k3_set_error("ibv_poll_cq (send) failed");
      return -1;
    }
    if (polled > 0) {
      if (wc.status != IBV_WC_SUCCESS) {
        k3_set_error("send completion status %d (%s)", wc.status,
                     ibv_wc_status_str(wc.status));
        return -1;
      }
      return 0;
    }
    if (k3_now_ms() >= deadline) {
      k3_set_error("k3rdma_wait_send timed out after %d ms", timeout_ms);
      return -1;
    }
    struct timespec nap = {0, 20000}; /* 20 us: keeps hot-path latency low; the pre-posted recv credit (see k3_rdma_transport.py) means waits almost always hit an already-pending completion */
    nanosleep(&nap, NULL);
  }
}

int k3rdma_wait_recv(int timeout_ms) {
  if (!g_ep.open || g_ep.recv_cq == NULL) {
    k3_set_error("k3rdma_wait_recv: endpoint not open");
    return -1;
  }
  const int64_t deadline = k3_now_ms() + (timeout_ms > 0 ? timeout_ms : 0);
  for (;;) {
    struct ibv_wc wc;
    memset(&wc, 0, sizeof(wc));
    int polled = ibv_poll_cq(g_ep.recv_cq, 1, &wc);
    if (polled < 0) {
      k3_set_error("ibv_poll_cq (recv) failed");
      return -1;
    }
    if (polled > 0) {
      if (wc.status != IBV_WC_SUCCESS) {
        k3_set_error("recv completion status %d (%s)", wc.status,
                     ibv_wc_status_str(wc.status));
        return -1;
      }
      return (int)wc.byte_len;
    }
    if (k3_now_ms() >= deadline) {
      k3_set_error("k3rdma_wait_recv timed out after %d ms", timeout_ms);
      return -1;
    }
    struct timespec nap = {0, 20000}; /* 20 us: keeps hot-path latency low; the pre-posted recv credit (see k3_rdma_transport.py) means waits almost always hit an already-pending completion */
    nanosleep(&nap, NULL);
  }
}

/*
 * Poll the recv CQ once for a completed receive.  Unlike
 * k3rdma_wait_recv() there is no deadline and no sleep: this is a single
 * non-blocking poll used by the server's idle loop so it can detect a dead
 * or reconnected client without blocking for the full recv window.
 *
 * Returns:
 *    1  a receive completed; *out_len is set to the received byte length.
 *    0  the recv CQ is empty (no completion pending).
 *   -1  poll error or a non-success completion status.
 */
int k3rdma_poll_recv(int* out_len) {
  if (!g_ep.open || g_ep.recv_cq == NULL) {
    k3_set_error("k3rdma_poll_recv: endpoint not open");
    return -1;
  }
  if (out_len == NULL) {
    k3_set_error("k3rdma_poll_recv: null out_len");
    return -1;
  }
  struct ibv_wc wc;
  memset(&wc, 0, sizeof(wc));
  int polled = ibv_poll_cq(g_ep.recv_cq, 1, &wc);
  if (polled < 0) {
    k3_set_error("ibv_poll_cq (recv) failed");
    return -1;
  }
  if (polled == 0) {
    return 0;
  }
  if (wc.status != IBV_WC_SUCCESS) {
    k3_set_error("recv completion status %d (%s)", wc.status,
                 ibv_wc_status_str(wc.status));
    return -1;
  }
  *out_len = (int)wc.byte_len;
  return 1;
}

void k3rdma_close(void) {
  if (g_ep.qp != NULL) {
    ibv_destroy_qp(g_ep.qp);
    g_ep.qp = NULL;
  }
  if (g_ep.send_cq != NULL) {
    ibv_destroy_cq(g_ep.send_cq);
    g_ep.send_cq = NULL;
  }
  if (g_ep.recv_cq != NULL) {
    ibv_destroy_cq(g_ep.recv_cq);
    g_ep.recv_cq = NULL;
  }
  if (g_ep.pd != NULL) {
    ibv_dealloc_pd(g_ep.pd);
    g_ep.pd = NULL;
  }
  if (g_ep.context != NULL) {
    ibv_close_device(g_ep.context);
    g_ep.context = NULL;
  }
  if (g_ep.dev_list != NULL) {
    ibv_free_device_list(g_ep.dev_list);
    g_ep.dev_list = NULL;
  }
  g_ep.open = 0;
}
