"""country_map.py — Fonte única de verdade: subconjunto -> (setor, país,
presente no 01_splits) e a atribuição balanceada país->Pi.

Origem dos dados: Tabelas 3 e 4 do dataset card oficial
(https://huggingface.co/datasets/ai-iot/EnergyBench), cruzadas com a
listagem real de `01_splits/Hourly/train/{Commercial,Residential}/*` do
usuário (8 de 67 subconjuntos ausentes — não processados/baixados;
decisão explícita do usuário: prosseguir sem eles, sem reprocessar).

Casos especiais decididos com o usuário:
  * ULE -> Reino Unido: o dataset card só diz "Cambridge"; confirmado
    por coincidência exata com a obs. comercial (8.783) já atribuída a
    "Reino Unido" na tabela LaTeX original do usuário, e por um
    repositório público condizente (EECi/Cambridge-Estates-Building-
    Energy-Archive). Confiança: razoável, não 100% oficial.
  * PES -> EUA: dataset card diz "USA, Europe" (multi-região); decisão
    do usuário foi simplificar para EUA.
  * Polônia: os DOIS subconjuntos (ECRG-Commercial, ECRG-Residential)
    estão ausentes -> Polônia tem ZERO dados reais processados; não
    aparece em nenhum balde.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Subset:
    setor: str      # "Commercial" | "Residential" (== pasta em 01_splits)
    pais: str
    presente: bool  # False = está no "missing" do EnergyBench, sem dados


# nome_local (== nome da pasta em 01_splits/Hourly/{split}/<setor>/<nome>)
SUBSETS: dict[str, Subset] = {
    "BDG-2":            Subset("Commercial", "EUA", True),
    "Berkely":          Subset("Commercial", "EUA", True),
    "CLEMD":            Subset("Commercial", "Malásia", True),
    "CU-BEMS":          Subset("Commercial", "Tailândia", True),
    "DGS":              Subset("Commercial", "EUA", True),
    "Enernoc":          Subset("Commercial", "EUA", True),
    "EWELD":            Subset("Commercial", "China", True),
    "HB":               Subset("Commercial", "EUA", True),
    "IBlend":           Subset("Commercial", "Índia", True),
    "IOT":              Subset("Commercial", "Emirados Árabes", True),
    "IPC-Commercial":   Subset("Commercial", "China", True),
    "NEST-Commercial":  Subset("Commercial", "Suíça", True),
    "PSS":              Subset("Commercial", "África do Sul", True),
    "RKP":              Subset("Commercial", "Portugal", True),
    "SEWA":             Subset("Commercial", "Emirados Árabes", True),
    "SKC":              Subset("Commercial", "Coreia do Sul", True),
    "UCIE":             Subset("Commercial", "Portugal", True),
    "ULE":              Subset("Commercial", "Reino Unido", True),
    "UNICON":           Subset("Commercial", "Austrália", True),
    "ECRG-Commercial":  Subset("Commercial", "Polônia", False),

    "AMPD":             Subset("Residential", "Canadá", True),
    "BTS":              Subset("Residential", "Austrália", True),
    "CEEW":             Subset("Residential", "Índia", True),
    "DCB":              Subset("Residential", "Japão", True),
    "DEDDIAG":          Subset("Residential", "Alemanha", True),
    "DESM":             Subset("Residential", "França", True),
    "DTH":              Subset("Residential", "Eslováquia", True),
    "ECCC":             Subset("Residential", "Portugal", True),
    "ECWM":             Subset("Residential", "México", True),
    "ENERTALK":         Subset("Residential", "Coreia do Sul", True),
    "fIEECe":           Subset("Residential", "EUA", True),
    "GoiEner":          Subset("Residential", "Espanha", True),
    "GREEND":           Subset("Residential", "Itália/Áustria", True),
    "HONDA-Smart-Home": Subset("Residential", "EUA", True),
    "HSG":              Subset("Residential", "Alemanha", True),
    "HUE":              Subset("Residential", "Canadá", True),
    "iFlex":            Subset("Residential", "Noruega", True),
    "IHEPC":            Subset("Residential", "França", True),
    "IPC-Residential":  Subset("Residential", "China", True),
    "IRH":              Subset("Residential", "Irlanda", True),
    "LCL":              Subset("Residential", "Reino Unido", True),
    "LEC":              Subset("Residential", "Portugal", True),
    "MFRED":            Subset("Residential", "EUA", True),
    "MIHEC":            Subset("Residential", "África do Sul", True),
    "NDB":              Subset("Residential", "Reino Unido", True),
    "NEST-Residential": Subset("Residential", "Suíça", True),
    "Norwegian":        Subset("Residential", "Noruega", True),
    "PES":              Subset("Residential", "EUA", True),          # decisão
    "Plegma":           Subset("Residential", "Grécia", True),
    "Prayas":           Subset("Residential", "Índia", True),
    "REED":             Subset("Residential", "Costa Rica", True),
    "REFIT":            Subset("Residential", "Reino Unido", True),
    "RHC":              Subset("Residential", "Canadá", True),
    "RSL":              Subset("Residential", "Sri Lanka", True),
    "SFAC":             Subset("Residential", "China", True),
    "SFHG":             Subset("Residential", "Alemanha", True),
    "SGSC":             Subset("Residential", "Austrália", True),
    "SMART-Star":       Subset("Residential", "EUA", True),
    "SRSA":             Subset("Residential", "África do Sul", True),
    "WED":              Subset("Residential", "México", True),
    "METER":            Subset("Residential", "Reino Unido", False),
    "SAVE":             Subset("Residential", "Reino Unido", False),
    "HES":              Subset("Residential", "Reino Unido", False),
    "UKST":             Subset("Residential", "Reino Unido", False),
    "NEEA":             Subset("Residential", "EUA", False),
    "ECRG-Residential": Subset("Residential", "Polônia", False),
    "NESEMP":           Subset("Residential", "Reino Unido", False),
}

# Atribuição país -> Pi, FIXA (balanceamento guloso por volume real,
# calculado e conferido em 2026-08-05; hardcoded para ser auditável e
# reprodutível — não recalculada a cada execução).
PI_BUCKETS: dict[int, list[str]] = {
    1: ["Espanha"],
    2: ["Austrália"],
    3: ["Reino Unido"],
    4: ["EUA", "China", "Eslováquia", "Alemanha", "Canadá",
        "Itália/Áustria", "Japão", "Irlanda", "Suíça", "México", "Malásia"],
    5: ["Noruega", "Portugal", "Sri Lanka", "Índia", "Tailândia",
        "África do Sul", "Coreia do Sul", "Grécia", "França",
        "Emirados Árabes", "Costa Rica"],
}

# Teto de janelas por Pi (cartão de 64GB, ~45GB livres medidos via df -h,
# margem de segurança de 3GB -> orçamento de 42GB; bytes/janela=4469,
# calculado do esquema real do shard). Só a Espanha (Pi1) excede isso.
CAP_JANELAS_POR_PI = 10_091_558


def groups_for_country(pais: str) -> list[str]:
    """Devolve os caminhos de grupo (<setor>/<subconjunto>) — o mesmo
    formato usado pelo --group do pipeline — para um país, só com
    subconjuntos PRESENTES."""
    return [f"{sub.setor}/{nome}" for nome, sub in SUBSETS.items()
            if sub.pais == pais and sub.presente]


def groups_for_pi(pi: int) -> list[str]:
    grupos = []
    for pais in PI_BUCKETS[pi]:
        grupos += groups_for_country(pais)
    return grupos


def pais_do_pi(pi: int) -> list[str]:
    return PI_BUCKETS[pi]


if __name__ == "__main__":
    # sanidade: nenhum país fora dos 5 baldes, nenhuma duplicata
    todos_baldes = [p for ps in PI_BUCKETS.values() for p in ps]
    todos_subset = {s.pais for s in SUBSETS.values() if s.presente}
    assert len(todos_baldes) == len(set(todos_baldes)), "país duplicado entre baldes"
    faltando = todos_subset - set(todos_baldes)
    sobrando = set(todos_baldes) - todos_subset
    print(f"Países com dado presente: {len(todos_subset)}")
    print(f"Países nos baldes: {len(todos_baldes)}")
    print(f"Sem balde (bug se não-vazio): {faltando}")
    print(f"Em balde mas sem dado presente (bug se não-vazio): {sobrando}")
    for pi in range(1, 6):
        g = groups_for_pi(pi)
        print(f"\nPi{pi} ({pais_do_pi(pi)}): {len(g)} grupos")
        for x in g:
            print(f"   {x}")