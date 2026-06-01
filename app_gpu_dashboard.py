import csv
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
PCAP_DIR = Path(os.environ.get("PCAP_DIR", str(BASE_DIR / "data" / "pcaps")))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(BASE_DIR / "data" / "results")))

PCAP_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="PAD Dashboard",
    page_icon="GPU",
    layout="wide",
)

# Persistent log area in the sidebar
if "app_logs" not in st.session_state:
    st.session_state["app_logs"] = ""
log_exp = st.sidebar.expander("Logs", expanded=True)
log_output = log_exp.empty()
LOG_PATH = RESULTS_DIR / "log.txt"


def run_command(command):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def run_python_command(args):
    """
    Run a Python command and return (exit_code, output).
    Non-streaming convenience wrapper.
    """
    cmd = [sys.executable] + args
    completed = subprocess.run(cmd, capture_output=True, text=True)
    output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    return completed.returncode, output.strip()


def run_python_command_stream(args, out_container=None, log_file=None):
    """
    Run a Python command and stream stdout/stderr line-by-line into
    a Streamlit container (out_container). Returns (exit_code, full_output).
    Also appends output to `st.session_state['app_logs']` and to `log_file` if provided.
    If out_container is None, collects output but still writes to log_file/session_state.
    """
    cmd = [sys.executable] + args
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    full = []

    # prepare logfile handle if requested
    lf = None
    try:
        if log_file:
            lf = open(log_file, "a", encoding="utf-8")

        text_buf = st.session_state.get("app_logs", "")

        for line in proc.stdout:
            full.append(line)
            text_buf += line

            # update session logs
            st.session_state["app_logs"] = text_buf

            # write to logfile
            if lf:
                lf.write(line)
                lf.flush()

            # stream to UI container if provided
            if out_container is not None:
                out_container.code(text_buf, language="text")

        proc.wait()

    except Exception as e:
        try:
            proc.kill()
        except Exception:
            pass
        err = f"\n[ERROR] {e}\n"
        full.append(err)
        text_buf += err
        st.session_state["app_logs"] = text_buf
        if lf:
            lf.write(err)
            lf.flush()
        if out_container is not None:
            out_container.code(text_buf, language="text")

    finally:
        if lf:
            lf.close()

    return proc.returncode, "".join(full).strip()


def format_gb(value_bytes):
    return round(value_bytes / (1024 ** 3), 2)


def list_pcap_files():
    files = []
    for ext in ("*.pcap", "*.pcapng"):
        files.extend(PCAP_DIR.glob(ext))
    return sorted(files)


def collect_host_info():
    cpu_freq = psutil.cpu_freq()
    virtual_memory = psutil.virtual_memory()
    disk_usage = psutil.disk_usage("/")

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpus": psutil.cpu_count(logical=True) or 0,
        "physical_cpus": psutil.cpu_count(logical=False) or 0,
        "cpu_percent": psutil.cpu_percent(interval=0.3),
        "cpu_freq_current": round(cpu_freq.current, 0) if cpu_freq else None,
        "memory_total_gb": format_gb(virtual_memory.total),
        "memory_used_gb": format_gb(virtual_memory.used),
        "memory_available_gb": format_gb(virtual_memory.available),
        "disk_total_gb": format_gb(disk_usage.total),
        "disk_used_gb": format_gb(disk_usage.used),
        "disk_free_gb": format_gb(disk_usage.free),
    }


def collect_gpu_info():
    if shutil.which("nvidia-smi") is None:
        return {
            "available": False,
            "rows": [],
            "raw_list": None,
            "raw_topology": None,
        }

    query = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.free,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]
    output = run_command(query)
    rows = []

    if output:
        reader = csv.reader(output.splitlines())
        for index, row in enumerate(reader):
            if len(row) < 10:
                continue
            rows.append(
                {
                    "gpu_index": index,
                    "name": row[0].strip(),
                    "driver_version": row[1].strip(),
                    "memory_total_gb": round(float(row[2]) / 1024, 2),
                    "memory_free_gb": round(float(row[3]) / 1024, 2),
                    "memory_used_gb": round(float(row[4]) / 1024, 2),
                    "util_gpu_pct": float(row[5]),
                    "util_mem_pct": float(row[6]),
                    "temperature_c": float(row[7]),
                }
            )

    return {
        "available": True,
        "rows": rows,
        "raw_list": run_command(["nvidia-smi", "-L"]),
        "raw_topology": run_command(["nvidia-smi", "topo", "-m"]),
    }


st.title("PAD Dashboard: Upload, Treino CPU e Autoencoder GPU")
st.caption(f"Snapshot: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption(f"PCAP volume: {PCAP_DIR} | Resultados: {RESULTS_DIR}")
st.info(
    "A interface completa tem 7 abas: Hardware, PCAPs, Treino CPU, Treino Federado, Autoencoder GPU, Métodos e Charts. "
    "Se preferir, use a seção abaixo para upload e execução rápida sem trocar de aba."
)

with st.expander("Upload e Processamento Rápido", expanded=True):
    quick_uploaded = st.file_uploader(
        "Upload rápido de PCAP/PCAPNG",
        type=["pcap", "pcapng"],
        accept_multiple_files=True,
        key="quick_upload",
    )

    if st.button("Salvar upload rápido", width="stretch", key="quick_save_button"):
        if not quick_uploaded:
            st.warning("Nenhum arquivo selecionado no upload rápido.")
        else:
            for item in quick_uploaded:
                target = PCAP_DIR / item.name
                with open(target, "wb") as f:
                    f.write(item.getbuffer())
            st.success(f"{len(quick_uploaded)} arquivo(s) salvo(s) em {PCAP_DIR}.")

    quick_files = list_pcap_files()
    st.write(f"Arquivos detectados no volume: {len(quick_files)}")

    col_q1, col_q2, col_q3 = st.columns(3)
    quick_cpu_out = col_q1.text_input("Saída CPU", value="cpu_results.pkl", key="quick_cpu_out")
    quick_gpu_out = col_q2.text_input("Saída GPU", value="gpu_results.pkl", key="quick_gpu_out")
    quick_chart_base = col_q3.text_input("Nome do chart", value="cpu_gpu_comparison", key="quick_chart_base")

    cqa, cqb, cqc = st.columns(3)
    if cqa.button("Executar CPU (Isolation Forest)", width="stretch", key="quick_run_cpu"):
        if not quick_files:
            st.error("Nenhum PCAP disponível para treino CPU.")
        else:
            args = [
                "cpu_train.py",
                "--shards",
                *[str(p) for p in quick_files[:4]],
                "--outdir",
                str(RESULTS_DIR),
                "--estimators",
                "200",
                "--contamination",
                "0.05",
                "--outfile",
                quick_cpu_out,
            ]
            with st.spinner("Executando CPU rápido..."):
                code, output = run_python_command_stream(args, out_container=log_output)
            if code == 0:
                st.success("Treino CPU rápido concluído.")
            else:
                st.error(f"Falha no CPU rápido (exit code {code}).")

    if cqb.button("Executar GPU (Autoencoder)", width="stretch", key="quick_run_gpu"):
        if not quick_files:
            st.error("Nenhum PCAP disponível para benchmark GPU.")
        else:
            args = [
                "gpu_train.py",
                "--shards",
                *[str(p) for p in quick_files[:4]],
                "--outdir",
                str(RESULTS_DIR),
                "--outfile",
                quick_gpu_out,
                "--epochs",
                "20",
                "--batch-size",
                "1024",
                "--latent-dim",
                "16",
                "--anom-percentile",
                "95.0",
            ]
            with st.spinner("Executando GPU rápido..."):
                code, output = run_python_command_stream(args, out_container=log_output)
            if code == 0:
                st.success("Benchmark GPU rápido concluído.")
            else:
                st.error(f"Falha no GPU rápido (exit code {code}).")

    if cqc.button("Gerar chart CPU x GPU", width="stretch", key="quick_run_chart"):
        args = [
            "plots_cpu_gpu_compare.py",
            "--cpu-results",
            str(RESULTS_DIR / quick_cpu_out),
            "--gpu-results",
            str(RESULTS_DIR / quick_gpu_out),
            "--outdir",
            str(RESULTS_DIR),
            "--basename",
            quick_chart_base,
        ]
        with st.spinner("Gerando chart rápido..."):
            code, output = run_python_command_stream(args, out_container=log_output)
        if code == 0:
            st.success("Chart rápido gerado.")
        else:
            st.error(f"Falha ao gerar chart rápido (exit code {code}).")

    quick_png = RESULTS_DIR / f"{quick_chart_base}.png"
    if quick_png.exists():
        st.image(str(quick_png), caption=f"Preview: {quick_png}")

tab_hw, tab_data, tab_cpu, tab_fed, tab_gpu, tab_methods, tab_charts = st.tabs(
    ["Hardware", "PCAPs", "Treino CPU", "Treino Federado", "Autoencoder GPU", "Métodos", "Charts"]
)

with tab_hw:
    host = collect_host_info()
    gpu = collect_gpu_info()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Logical CPUs", host["logical_cpus"])
    col2.metric("CPU usage", f"{host['cpu_percent']:.1f}%")
    col3.metric("RAM usada", f"{host['memory_used_gb']:.2f} GB")
    col4.metric("Disco usado", f"{host['disk_used_gb']:.2f} GB")

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "hostname": host["hostname"],
                    "platform": host["platform"],
                    "python": host["python"],
                    "physical_cpus": host["physical_cpus"],
                    "cpu_freq_mhz": host["cpu_freq_current"],
                    "memory_total_gb": host["memory_total_gb"],
                    "memory_available_gb": host["memory_available_gb"],
                    "disk_total_gb": host["disk_total_gb"],
                    "disk_free_gb": host["disk_free_gb"],
                }
            ]
        ),
        width="stretch",
    )

    st.subheader("Inventário de GPU")
    if not gpu["available"]:
        st.warning("nvidia-smi não encontrado no container.")
    elif gpu["rows"]:
        gpu_df = pd.DataFrame(gpu["rows"])
        st.dataframe(gpu_df, width="stretch")
        st.bar_chart(gpu_df.set_index("name")[["memory_used_gb", "memory_free_gb", "util_gpu_pct"]])
    else:
        st.info("GPU habilitada, mas sem linhas retornadas por nvidia-smi.")

    with st.expander("Saída bruta do nvidia-smi"):
        st.code(gpu["raw_list"] or "Sem saída", language="text")
        st.code(gpu["raw_topology"] or "Sem saída", language="text")

with tab_data:
    st.subheader("Upload de arquivos PCAP/PCAPNG")
    st.info("Para 4 arquivos de 1.25GB, o upload pode demorar e depender da rede/browser.")

    uploaded = st.file_uploader(
        "Selecione os arquivos",
        type=["pcap", "pcapng"],
        accept_multiple_files=True,
    )

    if st.button("Salvar arquivos no volume", width="stretch"):
        if not uploaded:
            st.warning("Nenhum arquivo selecionado.")
        else:
            for item in uploaded:
                target = PCAP_DIR / item.name
                with open(target, "wb") as f:
                    f.write(item.getbuffer())
            st.success(f"{len(uploaded)} arquivo(s) salvo(s) em {PCAP_DIR}.")

    files = list_pcap_files()
    if files:
        listing = [
            {
                "arquivo": p.name,
                "tamanho_gb": round(p.stat().st_size / (1024 ** 3), 3),
                "caminho": str(p),
            }
            for p in files
        ]
        st.dataframe(pd.DataFrame(listing), width="stretch")
    else:
        st.warning("Nenhum PCAP no volume ainda.")

with tab_cpu:
    st.subheader("Treinamento CPU com Isolation Forest")
    files = list_pcap_files()
    selected_cpu = st.multiselect(
        "Selecione os shards para CPU",
        options=[str(p) for p in files],
        default=[str(p) for p in files[:4]],
    )
    c1, c2 = st.columns(2)
    estimators = c1.number_input("N estimators", min_value=50, max_value=1000, value=200, step=50)
    contamination = c2.slider("Contamination", min_value=0.001, max_value=0.3, value=0.05, step=0.001)
    cpu_outfile = st.text_input("Arquivo de saída CPU", value="cpu_results.pkl")

    if st.button("Rodar treino CPU", width="stretch"):
        if not selected_cpu:
            st.error("Selecione ao menos um arquivo PCAP.")
        else:
            args = [
                "cpu_train.py",
                "--shards",
                *selected_cpu,
                "--outdir",
                str(RESULTS_DIR),
                "--estimators",
                str(estimators),
                "--contamination",
                str(contamination),
                "--outfile",
                cpu_outfile,
            ]
            with st.spinner("Executando treino CPU..."):
                code, output = run_python_command_stream(args, out_container=log_output)
            if code == 0:
                st.success("Treino CPU concluído.")
            else:
                st.error(f"Treino CPU falhou (exit code {code}).")

with tab_fed:
    st.subheader("Treinamento federado com Isolation Forest")
    files = list_pcap_files()
    selected_fed = st.multiselect(
        "Selecione os shards para federado",
        options=[str(p) for p in files],
        default=[str(p) for p in files[:4]],
    )
    f1, f2, f3 = st.columns(3)
    fed_rounds = f1.number_input("Rounds", min_value=1, max_value=20, value=6, step=1)
    fed_workers = f2.number_input("Workers", min_value=1, max_value=32, value=4, step=1)
    fed_estimators = f3.number_input("N trees", min_value=50, max_value=1000, value=200, step=50)
    fed_contamination = st.slider("Contamination", min_value=0.001, max_value=0.3, value=0.05, step=0.001)
    fed_outfile = st.text_input("Arquivo de saída federado", value="cpu_federated_results.pkl")

    if st.button("Rodar treino federado", width="stretch"):
        if not selected_fed:
            st.error("Selecione ao menos um arquivo PCAP.")
        else:
            args = [
                "federated_train.py",
                "--shards",
                *selected_fed,
                "--outdir",
                str(RESULTS_DIR),
                "--rounds",
                str(fed_rounds),
                "--workers",
                str(fed_workers),
                "--estimators",
                str(fed_estimators),
                "--contamination",
                str(fed_contamination),
                "--outfile",
                fed_outfile,
            ]
            with st.spinner("Executando treino federado..."):
                code, output = run_python_command_stream(args, out_container=log_output)
            if code == 0:
                st.success("Treino federado concluído.")
            else:
                st.error(f"Treino federado falhou (exit code {code}).")

with tab_gpu:
    st.subheader("Benchmark GPU com Autoencoder")
    files = list_pcap_files()
    selected_gpu = st.multiselect(
        "Selecione os shards para GPU",
        options=[str(p) for p in files],
        default=[str(p) for p in files[:4]],
    )
    g1, g2, g3 = st.columns(3)
    gpu_epochs = g1.number_input("Épocas", min_value=1, max_value=200, value=20, step=1)
    gpu_batch = g2.number_input("Batch size", min_value=32, max_value=8192, value=1024, step=32)
    gpu_latent = g3.number_input("Latent dim", min_value=2, max_value=256, value=16, step=2)
    gpu_percentile = st.slider("Percentil de anomalia", min_value=50.0, max_value=99.9, value=95.0, step=0.1)
    gpu_outfile = st.text_input("Arquivo de saída GPU", value="gpu_results.pkl")

    if st.button("Rodar benchmark GPU", width="stretch"):
        if not selected_gpu:
            st.error("Selecione ao menos um arquivo PCAP.")
        else:
            args = [
                "gpu_train.py",
                "--shards",
                *selected_gpu,
                "--outdir",
                str(RESULTS_DIR),
                "--outfile",
                gpu_outfile,
                "--epochs",
                str(gpu_epochs),
                "--batch-size",
                str(gpu_batch),
                "--latent-dim",
                str(gpu_latent),
                "--anom-percentile",
                str(gpu_percentile),
            ]
            with st.spinner("Executando benchmark GPU..."):
                code, output = run_python_command_stream(args, out_container=log_output)
            if code == 0:
                st.success("Benchmark GPU concluído.")
            else:
                st.error(f"Benchmark GPU falhou (exit code {code}).")

with tab_methods:
    st.subheader("Pré-processamento e Execução Multi-métodos")

    files = list_pcap_files()
    selected = st.multiselect(
        "Selecione shards para pré-processamento (cache)", options=[str(p) for p in files], default=[str(p) for p in files[:4]]
    )
    cache_name = st.text_input("Nome do arquivo de cache", value="features_cache.pkl")
    if st.button("Gerar cache de features (pré-processar)", width="stretch"):
        if not selected:
            st.error("Selecione ao menos um shard para pré-processamento.")
        else:
            args = [
                "preprocess_features.py",
                "--shards",
                *selected,
                "--outdir",
                str(RESULTS_DIR),
                "--cache-file",
                cache_name,
            ]
            with st.spinner("Executando pré-processamento... Isso pode demorar:"):
                code, output = run_python_command_stream(args, out_container=log_output)
            if code == 0:
                st.success("Cache gerado com sucesso.")
            else:
                st.error(f"Falha ao gerar cache (exit code {code}).")

    st.markdown("---")
    st.write("Escolha métodos para executar sobre o cache (após gerar cache)")
    methods = st.multiselect(
        "Métodos",
        options=["IsolationForest (CPU)", "IsolationForest Federado", "Autoencoder (GPU)"],
        default=["IsolationForest (CPU)", "IsolationForest Federado", "Autoencoder (GPU)"],
    )
    out_base = st.text_input("Prefixo de saída para resultados", value="run_")
    run_btn = st.button("Executar métodos selecionados", width="stretch")
    if run_btn:
        cache_path = RESULTS_DIR / cache_name
        if not cache_path.exists():
            st.error(f"Arquivo de cache não encontrado: {cache_path}. Gere o cache primeiro.")
        else:
            results_files = []
            for m in methods:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if m == "IsolationForest (CPU)":
                    outname = f"{out_base}if_cpu_{timestamp}.pkl"
                    args = [
                        "cpu_train.py",
                        "--cache-file",
                        str(cache_path),
                        "--outdir",
                        str(RESULTS_DIR),
                        "--outfile",
                        outname,
                        "--estimators",
                        "200",
                        "--contamination",
                        "0.05",
                    ]
                elif m == "IsolationForest Federado":
                    outname = f"{out_base}if_fed_{timestamp}.pkl"
                    args = [
                        "federated_train.py",
                        "--cache-file",
                        str(cache_path),
                        "--outdir",
                        str(RESULTS_DIR),
                        "--outfile",
                        outname,
                        "--rounds",
                        "6",
                        "--workers",
                        "4",
                        "--estimators",
                        "200",
                        "--contamination",
                        "0.05",
                    ]
                elif m == "Autoencoder (GPU)":
                    outname = f"{out_base}ae_gpu_{timestamp}.pkl"
                    args = [
                        "gpu_train.py",
                        "--cache-file",
                        str(cache_path),
                        "--outdir",
                        str(RESULTS_DIR),
                        "--outfile",
                        outname,
                        "--epochs",
                        "20",
                        "--batch-size",
                        "1024",
                        "--latent-dim",
                        "16",
                        "--anom-percentile",
                        "95.0",
                    ]
                else:
                    continue

                with st.spinner(f"Executando {m} ..."):
                    code, output = run_python_command_stream(args, out_container=log_output)
                if code == 0:
                    st.success(f"{m} concluído -> {outname}")
                    results_files.append(str(RESULTS_DIR / outname))
                else:
                    st.error(f"{m} falhou (exit code {code}).")
                st.code(output or "Sem saída", language="text")

            if results_files:
                st.write("Resultados gerados:")
                for f in results_files:
                    st.write(f)

with tab_charts:
    st.subheader("Geração de charts de comparação")
    cpu_file = st.text_input("Resultado CPU (.pkl)", value=str(RESULTS_DIR / "cpu_results.pkl"))
    gpu_file = st.text_input("Resultado GPU (.pkl)", value=str(RESULTS_DIR / "gpu_results.pkl"))
    base_name = st.text_input("Nome base do chart", value="cpu_gpu_comparison")

    if st.button("Gerar chart CPU x GPU", width="stretch"):
        args = [
            "plots_cpu_gpu_compare.py",
            "--cpu-results",
            cpu_file,
            "--gpu-results",
            gpu_file,
            "--outdir",
            str(RESULTS_DIR),
            "--basename",
            base_name,
        ]
        with st.spinner("Gerando chart..."):
                code, output = run_python_command_stream(args, out_container=log_output)
        if code == 0:
            st.success("Chart gerado com sucesso.")
        else:
            st.error(f"Falha ao gerar chart (exit code {code}).")

    png_path = RESULTS_DIR / f"{base_name}.png"
    if png_path.exists():
        st.image(str(png_path), caption=str(png_path))
