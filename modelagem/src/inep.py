"""Leitura das planilhas de Indicadores Educacionais do INEP (nível escola)."""
from pathlib import Path

import pandas as pd
from python_calamine import CalamineWorkbook

DATA = Path(__file__).resolve().parents[1] / "data"


def _header_row(path):
    """Índice da linha de códigos (a que começa com NU_ANO_CENSO)."""
    sheet = CalamineWorkbook.from_path(str(path)).get_sheet_by_index(0)
    for i, row in enumerate(sheet.to_python(nrows=30)):
        if row and str(row[0]).strip() == "NU_ANO_CENSO":
            return i
    raise ValueError(f"Linha de cabeçalho não encontrada em {path}")


def to_num(s):
    """Converte para número; '--', '*' e afins viram NaN."""
    return pd.to_numeric(s.astype(str).str.strip().str.replace(",", ".", regex=False),
                         errors="coerce")


def read_inep(prefix, year, uf="SP", redes=("Municipal", "Estadual")):
    """Lê a planilha `{prefix}_{year}.xlsx`, remove o rodapé e filtra UF e redes."""
    path = DATA / f"{prefix}_{year}.xlsx"
    df = pd.read_excel(path, header=_header_row(path), dtype=str, engine="calamine")
    df.columns = [str(c).strip() for c in df.columns]
    df = df[df["CO_ENTIDADE"].astype(str).str.strip().str.fullmatch(r"\d+")].copy()
    for c in ["SG_UF", "CO_MUNICIPIO", "CO_ENTIDADE", "NO_CATEGORIA", "NO_DEPENDENCIA"]:
        df[c] = df[c].astype(str).str.strip()
    df = df[(df["SG_UF"] == uf) & df["NO_DEPENDENCIA"].isin(redes)]
    return df.reset_index(drop=True)
