"""
feature_extractor.py
Extração de features de fluxo (5-tupla) a partir de arquivos pcapng.
Usado por cada worker no pipeline federado.

Dependências:
    pip install pyshark pandas numpy
"""

import pyshark
import pandas as pd
import numpy as np
from pathlib import Path
import hashlib
import warnings
warnings.filterwarnings("ignore")

try:
    from pyshark.capture.capture import TSharkCrashException
except Exception:
    TSharkCrashException = Exception

# ── Features extraídas por fluxo ──────────────────────────────
FEATURE_COLS = [
    "duration", "proto",
    "src_port", "dst_port",
    "pkt_count", "byte_count",
    "mean_pkt_size", "std_pkt_size",
    "mean_iat", "std_iat", "min_iat", "max_iat",
    "flag_syn", "flag_fin", "flag_rst", "flag_psh",
    "fwd_pkt_count", "bwd_pkt_count", "fwd_byte_ratio",
    "is_port_well_known",   # dst_port < 1024
    "is_ephemeral_src",     # src_port > 49151
    "bytes_per_pkt",
]


def _anonymize_ip(ip: str) -> str:
    """Substitui IP por hash SHA-256 truncado (10 chars) — LGPD."""
    return hashlib.sha256(ip.encode()).hexdigest()[:10]


def extract_flows(pcap_path: str, anonymize: bool = True) -> pd.DataFrame:
    """
    Lê um arquivo pcapng e retorna um DataFrame com features
    agregadas por fluxo (5-tupla: src_ip, dst_ip, src_port,
    dst_port, proto).

    Args:
        pcap_path:  caminho para o arquivo .pcapng
        anonymize:  se True, substitui IPs por hash (recomendado)

    Returns:
        pd.DataFrame com FEATURE_COLS como colunas.
    """
    pcap_path = str(pcap_path)
    cap = pyshark.FileCapture(pcap_path, keep_packets=False)
    flows: dict = {}

    try:
        for pkt in cap:
            try:
                if not hasattr(pkt, "ip") or not hasattr(pkt, "transport_layer"):
                    continue
                proto = pkt.transport_layer
                src_ip = pkt.ip.src
                dst_ip = pkt.ip.dst
                sp = int(getattr(pkt[proto], "srcport", 0))
                dp = int(getattr(pkt[proto], "dstport", 0))
                ts = float(pkt.sniff_timestamp)
                size = int(pkt.length)

                # Anonimização
                if anonymize:
                    src_key = _anonymize_ip(src_ip)
                    dst_key = _anonymize_ip(dst_ip)
                else:
                    src_key, dst_key = src_ip, dst_ip

                key = (src_key, dst_key, sp, dp, proto)

                if key not in flows:
                    flows[key] = dict(
                        t_start=ts, t_last=ts,
                        pkts=[size], iats=[],
                        flags_syn=0, flags_fin=0, flags_rst=0, flags_psh=0,
                        fwd_bytes=0, bwd_bytes=0,
                        fwd_pkts=0, bwd_pkts=0,
                        total_bytes=0,
                    )
                else:
                    f = flows[key]
                    f["iats"].append(ts - f["t_last"])
                    f["t_last"] = ts
                    f["pkts"].append(size)

                f = flows[key]
                f["total_bytes"] += size
                f["fwd_pkts"] += 1
                f["fwd_bytes"] += size

                # Flags TCP
                if proto == "TCP":
                    try:
                        flags = int(pkt.tcp.flags, 16)
                        f["flags_syn"] += int(bool(flags & 0x02))
                        f["flags_fin"] += int(bool(flags & 0x01))
                        f["flags_rst"] += int(bool(flags & 0x04))
                        f["flags_psh"] += int(bool(flags & 0x08))
                    except Exception:
                        pass

            except AttributeError:
                continue
    finally:
        try:
            cap.close()
        except TSharkCrashException as exc:
            warnings.warn(
                f"TShark crashed while closing capture for {pcap_path}: {exc}. "
                "Returning the flows parsed before the crash.",
                RuntimeWarning,
            )
        except Exception as exc:
            warnings.warn(
                f"Unexpected error while closing capture for {pcap_path}: {exc}. "
                "Returning the flows parsed before the close failure.",
                RuntimeWarning,
            )

    records = []
    for (src, dst, sp, dp, proto), f in flows.items():
        pkts = np.array(f["pkts"])
        iats = np.array(f["iats"]) if f["iats"] else np.array([0.0])
        total = f["total_bytes"]
        fwd_b = f["fwd_bytes"]

        records.append({
            "duration":          f["t_last"] - f["t_start"],
            "proto":             1 if proto == "TCP" else 0,
            "src_port":          sp,
            "dst_port":          dp,
            "pkt_count":         len(pkts),
            "byte_count":        total,
            "mean_pkt_size":     float(pkts.mean()),
            "std_pkt_size":      float(pkts.std()),
            "mean_iat":          float(iats.mean()),
            "std_iat":           float(iats.std()),
            "min_iat":           float(iats.min()),
            "max_iat":           float(iats.max()),
            "flag_syn":          f["flags_syn"],
            "flag_fin":          f["flags_fin"],
            "flag_rst":          f["flags_rst"],
            "flag_psh":          f["flags_psh"],
            "fwd_pkt_count":     f["fwd_pkts"],
            "bwd_pkt_count":     f["bwd_pkts"],
            "fwd_byte_ratio":    fwd_b / total if total > 0 else 0.0,
            "is_port_well_known": int(dp < 1024),
            "is_ephemeral_src":   int(sp > 49151),
            "bytes_per_pkt":      total / len(pkts) if len(pkts) > 0 else 0.0,
        })

    df = pd.DataFrame(records, columns=FEATURE_COLS)
    return df.fillna(0)


if __name__ == "__main__":
    import sys, time
    if len(sys.argv) < 2:
        print("Uso: python feature_extractor.py <arquivo.pcapng>")
        sys.exit(1)
    t0 = time.perf_counter()
    df = extract_flows(sys.argv[1])
    elapsed = time.perf_counter() - t0
    print(f"Fluxos extraídos : {len(df):,}")
    print(f"Features          : {len(df.columns)}")
    print(f"Tempo             : {elapsed:.2f}s")
    print(df.describe().to_string())
