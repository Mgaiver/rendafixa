import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- Configurações da Página e Funções ---
st.set_page_config(layout="wide")

# --- CORREÇÃO 1: Dicionário de Mapeamento Atualizado ---
# Mapeia todos os nomes de colunas possíveis para um nome PADRÃO.
COLUMN_MAPPING = {
    'Ativo': 'Ativo', 'Vencimento': 'Vencimento', 'Tax.Máx': 'Taxa Máxima',
    'Tax.Mín': 'Taxa Mínima', 'Taxa Min/Máx': 'Taxa Contratada Texto',
    'GrossUp Tax.Máx': 'GrossUp Máximo', 'GrossUp Tax.Mín': 'GrossUp Mínimo',
    'Gross Up': 'Gross Up Texto', 'P.U': 'Preço Unitário', 'Qtd. Disp.': 'Qtd Disponível',
    'Rating': 'Rating', 'ROA Escritório': 'ROA', 'ROA E. Aprox.': 'ROA',
    'Risco': 'Risco', 'Público Alvo': 'Público Alvo', 'Público': 'Público Alvo',
    'Isento': 'Isento IR', 'Incentivada': 'Isento IR', 'Emissor': 'Emissor',
    'Indexador': 'Indexador', 'Ticker': 'Ticker'
}

def formatar_prazo_humanizado(dias):
    if pd.isna(dias) or dias < 0: return "N/A"
    if dias == 0: return "Hoje"
    anos, dias_rest = divmod(int(dias), 365)
    meses, dias_finais = divmod(dias_rest, 30)
    partes = []
    if anos > 0: partes.append(f"{anos} ano{'s' if anos > 1 else ''}")
    if meses > 0: partes.append(f"{meses} {'meses' if meses > 1 else 'mês'}")
    if dias_finais > 0: partes.append(f"{dias_finais} dia{'s' if dias_finais > 1 else ''}")
    return ", ".join(partes) if partes else "Menos de 1 mês"

@st.cache_data
def load_and_consolidate_data(uploaded_files):
    all_dfs = []
    def get_asset_type(filename):
        fn = filename.lower()
        if 'bancaria' in fn: return 'Emissão Bancária'
        if 'privado' in fn: return 'Crédito Privado'
        if 'publico' in fn: return 'Títulos Públicos'
        if 'financeira' in fn: return 'Letras Financeiras'
        return 'Outros'

    for file in uploaded_files:
        df = pd.read_excel(file)
        df['Tipo de Ativo'] = get_asset_type(file.name)
        df.rename(columns=COLUMN_MAPPING, inplace=True)
        all_dfs.append(df)
    
    if not all_dfs: return pd.DataFrame()
    master_df = pd.concat(all_dfs, ignore_index=True)

    # --- CORREÇÃO 2: Lógica de conversão numérica mais robusta ---
    cols_to_numeric = {'ROA': 'ROA_num', 'Taxa Máxima': 'Taxa Máxima_num'}
    for col, new_col in cols_to_numeric.items():
        if col in master_df.columns:
            # Converte para string, troca vírgula por ponto, extrai o número e converte para float
            master_df[new_col] = pd.to_numeric(
                master_df[col].astype(str).str.replace(',', '.', regex=False).str.extract(r'(\d+\.?\d*)')[0],
                errors='coerce'
            )

    # Cria colunas de texto formatadas para exibição
    if 'GrossUp Mínimo' in master_df.columns and 'GrossUp Máximo' in master_df.columns:
        master_df['Gross Up'] = master_df['GrossUp Mínimo'].astype(str) + ' - ' + master_df['GrossUp Máximo'].astype(str)
    
    if 'Vencimento' in master_df.columns:
        master_df['Vencimento'] = pd.to_datetime(master_df['Vencimento'], errors='coerce')
        master_df['Vencimento Formatado'] = master_df['Vencimento'].dt.strftime('%d/%m/%Y')
        def calcular_prazo_carencia(row):
            hoje = datetime.now().date()
            if pd.isna(row['Vencimento']): return np.nan
            return max((row['Vencimento'].date() - hoje).days, 0)
        master_df['Prazo Carência (dias)'] = master_df.apply(calcular_prazo_carencia, axis=1)
        master_df['Prazo'] = master_df['Prazo Carência (dias)'].apply(formatar_prazo_humanizado)

    return master_df

def create_plot(df_filtered):
    plot_cols = ['Taxa Máxima_num', 'ROA_num', 'Prazo Carência (dias)']
    if df_filtered.empty or not all(c in df_filtered.columns for c in plot_cols):
        return None, pd.DataFrame()

    df_plot = df_filtered.dropna(subset=plot_cols)
    if df_plot.empty: return None, pd.DataFrame()

    tax_range = df_plot['Taxa Máxima_num'].max() - df_plot['Taxa Máxima_num'].min()
    roa_range = df_plot['ROA_num'].max() - df_plot['ROA_num'].min()
    df_plot['Taxa_norm'] = 0.5 if tax_range == 0 else (df_plot['Taxa Máxima_num'] - df_plot['Taxa Máxima_num'].min()) / tax_range
    df_plot['ROA_norm'] = 0.5 if roa_range == 0 else (df_plot['ROA_num'] - df_plot['ROA_num'].min()) / roa_range
    df_plot['Score'] = df_plot['Taxa_norm'] + df_plot['ROA_norm']
    top3 = df_plot.sort_values('Score', ascending=False).head(3)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.kdeplot(x=df_plot['Taxa Máxima_num'], y=df_plot['ROA_num'], cmap="YlOrRd", fill=True, thresh=0.05, alpha=0.3, ax=ax)
    scatter = ax.scatter(x=df_plot['Taxa Máxima_num'], y=df_plot['ROA_num'], c=df_plot['Prazo Carência (dias)'], cmap='RdYlGn_r', s=90, edgecolor='k', alpha=0.85)
    
    if 'Ativo' in top3.columns:
        for _, row in top3.iterrows():
            ax.text(row['Taxa Máxima_num'], row['ROA_num'], f" {row['Ativo']}", fontsize=9, weight='bold', color='navy', va='bottom')
            ax.scatter(row['Taxa Máxima_num'], row['ROA_num'], c='blue', s=200, marker='*', label='Top 3 Ativos', zorder=5)

    ax.set_xlabel('Taxa Máxima Contratada'); ax.set_ylabel('ROA (%)')
    ax.set_title('Análise Consolidada: Risco (ROA) vs. Retorno (Taxa)', fontsize=16)
    fig.colorbar(scatter, ax=ax, label='Prazo de Carência (dias)'); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels(); by_label = dict(zip(labels, handles))
    if by_label: ax.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    return fig, top3

# --- Interface do Aplicativo (sem grandes alterações) ---
st.title('Analisador Consolidado de Renda Fixa 📊')
# ... (o resto da interface continua igual)

st.sidebar.header('1. Carregue os Arquivos')
uploaded_files = st.sidebar.file_uploader(
    "Pode selecionar vários arquivos de uma vez",
    type=["xlsx", "xls"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Por favor, carregue um ou mais arquivos Excel para começar."); st.stop()

df_master = load_and_consolidate_data(uploaded_files)
df_filtered = df_master.copy()

st.sidebar.header('2. Filtre as Oportunidades')
with st.sidebar.expander("ℹ️ Como Usar o Aplicativo"):
    st.markdown("Use os filtros para refinar sua busca. O gráfico e as tabelas serão atualizados automaticamente.")
# ... (filtros continuam iguais)

# --- Exibição dos Resultados ---
st.header('Resultados Filtrados')
if df_filtered.empty:
    st.warning("Nenhum ativo encontrado com os filtros selecionados.")
else:
    fig, top3 = create_plot(df_filtered)
    
    # Mensagem de erro foi movida para ser mais específica
    if not fig:
        st.warning("Não foi possível gerar o gráfico. Verifique se os ativos filtrados possuem dados numéricos de 'ROA' e 'Taxa Máxima'.")
    else:
        st.pyplot(fig)

    # --- CORREÇÃO 3: Nomes de coluna para exibição atualizados ---
    cols_to_display = [
        'Ativo', 'Tipo de Ativo', 'Vencimento Formatado', 'Prazo', 
        'Taxa Contratada Texto', 'Gross Up', 'Preço Unitário', 'Qtd Disponível', 'Rating', 
        'ROA', 'Risco', 'Público Alvo', 'Isento IR', 'Emissor', 'Indexador', 'Ticker'
    ]
    available_cols = [col for col in cols_to_display if col in df_filtered.columns]

    st.subheader(f'Total de {len(df_filtered)} Ativos Encontrados')
    if available_cols:
        st.dataframe(df_filtered[available_cols])