"""Monta a base única do Radar Docente (uma linha = escola x ano-base).

Uso:  python src/build_dataset.py
  -> data/base_radar_docente.csv   (anos-base 2022-2024, com alvo: treino e teste)
  -> data/base_previsao_2025.csv   (ano-base 2025, sem alvo: previsão para 2026)
"""
import re

import pandas as pd

from inep import DATA, read_inep, to_num

ANOS_BASE = [2022, 2023, 2024]  # features de t, lag de t-1, alvo de t+1
ANO_PREVISAO = 2025             # features de 2025 -> risco previsto para 2026
LIMIAR_BAIXA = 2.0              # faixa "baixa regularidade" do INEP
ID_COLS = ["CO_ENTIDADE", "NO_ENTIDADE", "CO_MUNICIPIO", "NO_MUNICIPIO",
           "NO_DEPENDENCIA", "NO_CATEGORIA"]


def ird(ano):
    df = read_inep("IRD_ESCOLAS", ano)
    return df.assign(ird=to_num(df["EDU_BAS_CAT_0"]))[ID_COLS + ["ird"]]


def pct_baixa_municipio(ano):
    """% de escolas municipais e estaduais do município com IRD < 2 (a partir do IRD por escola)."""
    df = read_inep("IRD_ESCOLAS", ano)
    df["ird"] = to_num(df["EDU_BAS_CAT_0"])
    df = df.dropna(subset=["ird"])
    return (df.assign(baixa=df["ird"] < LIMIAR_BAIXA)
              .groupby("CO_MUNICIPIO")["baixa"].mean().mul(100)
              .rename("pct_baixa_mun_t").reset_index())


def indicadores(ano):
    """AFD, IED, ICG, ATU e Rendimento do ano, já reduzidos às features por etapa."""
    afd = read_inep("AFD_ESCOLAS", ano)
    afd = pd.DataFrame({"CO_ENTIDADE": afd["CO_ENTIDADE"],
                        "afd_g1_fun": to_num(afd["FUN_CAT_1"]),
                        "afd_g1_med": to_num(afd["MED_CAT_1"])})

    ied = read_inep("IED_ESCOLAS", ano)
    alto = lambda et: pd.concat([to_num(ied[f"{et}_CAT_{k}"]) for k in (4, 5, 6)],
                                axis=1).sum(axis=1, min_count=1)
    ied = pd.DataFrame({"CO_ENTIDADE": ied["CO_ENTIDADE"],
                        "ied_alto_fun": alto("FUN"), "ied_alto_med": alto("MED")})

    icg = read_inep("ICG_ESCOLAS", ano)
    icg = pd.DataFrame({"CO_ENTIDADE": icg["CO_ENTIDADE"],
                        "icg": pd.to_numeric(icg["COMPLEX"].str.extract(r"(\d)")[0],
                                             errors="coerce")})

    atu = read_inep("ATU_ESCOLAS", ano)
    atu = pd.DataFrame({"CO_ENTIDADE": atu["CO_ENTIDADE"],
                        "atu_fun": to_num(atu["FUN_CAT_0"]),
                        "atu_med": to_num(atu["MED_CAT_0"])})

    rend = read_inep("tx_rend_escolas", ano)
    rend = pd.DataFrame({"CO_ENTIDADE": rend["CO_ENTIDADE"],
                         "reprov_fun": to_num(rend["2_CAT_FUN"]),
                         "reprov_med": to_num(rend["2_CAT_MED"]),
                         "aband_fun": to_num(rend["3_CAT_FUN"]),
                         "aband_med": to_num(rend["3_CAT_MED"])})

    out = afd
    for d in (ied, icg, atu, rend):
        out = out.merge(d, on="CO_ENTIDADE", how="outer")
    return out


def inse():
    df = pd.read_excel(DATA / "INSE_2023_escolas.xlsx", dtype=str, engine="calamine")
    df = df[df["SG_UF"].str.strip() == "SP"]
    return pd.DataFrame({"CO_ENTIDADE": df["ID_ESCOLA"].str.strip(),
                         "inse_media": to_num(df["MEDIA_INSE"]),
                         "inse_classe": df["INSE_CLASSIFICACAO"].str.strip()})


RENOMEAR = {"CO_ENTIDADE": "co_escola", "NO_ENTIDADE": "no_escola",
            "CO_MUNICIPIO": "co_municipio", "NO_MUNICIPIO": "no_municipio",
            "NO_DEPENDENCIA": "rede", "NO_CATEGORIA": "localizacao"}
COLUNAS = ["ano_base", "co_escola", "no_escola", "co_municipio", "no_municipio",
           "rede", "localizacao", "ird_t", "ird_t_1", "delta_ird", "pct_baixa_mun_t",
           "afd_g1_fun", "afd_g1_med", "ied_alto_fun", "ied_alto_med", "icg",
           "atu_fun", "atu_med", "reprov_fun", "reprov_med", "aband_fun", "aband_med",
           "tem_fundamental", "tem_medio", "inse_media", "inse_classe"]


def features_do_ano(t, irds):
    """Features do ano t (lag de t-1) para as escolas com IRD em t e com EF e/ou EM."""
    df = irds[t].rename(columns={"ird": "ird_t"}).dropna(subset=["ird_t"])
    df = df.merge(irds[t - 1][["CO_ENTIDADE", "ird"]].rename(columns={"ird": "ird_t_1"}),
                  on="CO_ENTIDADE", how="left")
    df = df.merge(pct_baixa_municipio(t), on="CO_MUNICIPIO", how="left")
    df = df.merge(indicadores(t), on="CO_ENTIDADE", how="left")

    # Etapa oferecida = a escola tem turmas (ATU) ou rendimento naquela etapa
    etapa = lambda cols: df[cols].notna().any(axis=1).astype(int)
    df["tem_fundamental"] = etapa(["atu_fun", "reprov_fun", "aband_fun"])
    df["tem_medio"] = etapa(["atu_med", "reprov_med", "aband_med"])
    df = df[(df["tem_fundamental"] == 1) | (df["tem_medio"] == 1)]  # universo EF/EM

    df["ano_base"] = t
    df["delta_ird"] = df["ird_t"] - df["ird_t_1"]
    return df


def main():
    irds = {a: ird(a) for a in range(ANOS_BASE[0] - 1, ANO_PREVISAO + 1)}
    tab_inse = inse()

    linhas = []
    for t in ANOS_BASE:
        print(f"ano-base {t}...")
        df = features_do_ano(t, irds)
        df = df.merge(irds[t + 1][["CO_ENTIDADE", "ird"]].rename(columns={"ird": "ird_t1"}),
                      on="CO_ENTIDADE", how="left")
        df = df.dropna(subset=["ird_t1"])                    # IRD em t e em t+1
        df["alvo"] = (df["ird_t1"] < LIMIAR_BAIXA).astype(int)
        linhas.append(df)

    base = (pd.concat(linhas, ignore_index=True).merge(tab_inse, on="CO_ENTIDADE", how="left")
            .rename(columns=RENOMEAR)[COLUNAS + ["ird_t1", "alvo"]])
    out = DATA / "base_radar_docente.csv"
    base.to_csv(out, index=False, encoding="utf-8")
    print(out, base.shape)
    print(base.groupby("ano_base")["alvo"].agg(["size", "mean"]))

    # Features de 2025 (sem alvo): entrada do modelo para prever o risco de 2026
    print(f"ano-base {ANO_PREVISAO} (previsão)...")
    prev = (features_do_ano(ANO_PREVISAO, irds).merge(tab_inse, on="CO_ENTIDADE", how="left")
            .rename(columns=RENOMEAR)[COLUNAS])
    out = DATA / f"base_previsao_{ANO_PREVISAO}.csv"
    prev.to_csv(out, index=False, encoding="utf-8")
    print(out, prev.shape)


if __name__ == "__main__":
    main()
