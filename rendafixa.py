import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- Configurações da Página e Funções ---
st.set_page_config(layout="wide")

@st.cache_data
def load_and_process_data(uploaded_file):
    """Carrega o arquivo excel e aplica a limpeza e engenharia de features iniciais."""
    df = pd.read_excel(uploaded_file, sheet_name='Resultado')

    def clean_columns(df):
        if 'Tax.Máx' in df.columns:
            df['Tax.Máx'] = (
                df['Tax.Máx'].astype(str)
                .str.replace('% CDI', '', regex=False).str.replace(',', '.')
                .str.extract(r'(\d+\.?\d*)')[0].astype(float)
            )
        if 'ROA E. Aprox.' in df.columns:
            df['ROA E. Aprox.'] = (
                df['ROA E. Aprox.'].astype(str)
                .str.replace('%', '', regex=False).str.replace(',', '.')
                .str.extract(r'(\d+\.?\d*)')[0].astype(float)
            )
        
        required_cols = ['Tax.Máx', 'ROA E. Aprox.', 'Vencimento']
        existing_required_cols = [col for col in required_cols if col in df.columns]
        return df.dropna(subset=existing_required_cols)

    df = clean_columns(df)

    def calcular_prazo_carencia(row):
        hoje = datetime.now().date()
        try:
            venc = pd.to_datetime(row['Vencimento']).date()
        except:
            return np.nan
        carencia = str(row.get('Carência', '')).strip().lower()
        if carencia in ['nan', '', 'vencimento', 'no vencimento']:
            prazo = (venc - hoje).days
        else:
            try:
                prazo_num = int(''.join(filter(str.isdigit, carencia)))
                prazo = prazo_num
            except:
                prazo = (venc - hoje).days
        return max(prazo, 0)

    if 'Vencimento' in df.columns:
        df['Prazo Carência (dias)'] = df.apply(calcular_prazo_carencia, axis=1)
        df.dropna(subset=['Prazo Carência (dias)'], inplace=True)
        df['Prazo Carência (dias)'] = df['Prazo Carência (dias)'].astype(int)

    return df

def create_plot(df_filtered):
    """Cria o gráfico de dispersão com base no dataframe filtrado."""
    if df_filtered.empty or 'Tax.Máx' not in df_filtered.columns or 'ROA E. Aprox.' not in df_filtered.columns:
        return None, pd.DataFrame()

    tax_min, tax_max = df_filtered['Tax.Máx'].min(), df_filtered['Tax.Máx'].max()
    roa_min, roa_max = df_filtered['ROA E. Aprox.'].min(), df_filtered['ROA E. Aprox.'].max()
    
    df_filtered['Taxa_norm'] = 0.5 if (tax_max - tax_min) == 0 else (df_filtered['Tax.Máx'] - tax_min) / (tax_max - tax_min)
    df_filtered['ROA_norm'] = 0.5 if (roa_max - roa_min) == 0 else (df_filtered['ROA E. Aprox.'] - roa_min) / (roa_max - roa_min)
    df_filtered['Score'] = df_filtered['Taxa_norm'] + df_filtered['ROA_norm']

    ## CORREÇÃO 1: GRÁFICO - Ajustado o tamanho da figura para uma melhor proporção (mais largo)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    sns.kdeplot(
        x=df_filtered['Tax.Máx'], y=df_filtered['ROA E. Aprox.'],
        cmap="YlOrRd", fill=True, thresh=0.05, levels=100, alpha=0.3, ax=ax
    )
    norm = plt.Normalize(df_filtered['Prazo Carência (dias)'].min(), df_filtered['Prazo Carência (dias)'].max())
    scatter = ax.scatter(
        df_filtered['Tax.Máx'], df_filtered['ROA E. Aprox.'],
        c=df_filtered['Prazo Carência (dias)'], cmap='RdYlGn_r', s=90, edgecolor='k', alpha=0.85, norm=norm
    )
    
    top3 = df_filtered.sort_values('Score', ascending=False).head(3)
    if 'Ativo' in top3.columns:
        # Calcula um pequeno deslocamento para o texto não ficar sobre o ponto
        x_offset = (tax_max - tax_min) * 0.01 
        for _, row in top3.iterrows():
            ## CORREÇÃO 2: GRÁFICO - Texto deslocado para a direita para não cobrir a estrela
            ax.text(row['Tax.Máx'] + x_offset, row['ROA E. Aprox.'], f" {row['Ativo']}", 
                    fontsize=9, weight='bold', color='navy', va='center')
            ax.scatter(row['Tax.Máx'], row['ROA E. Aprox.'], c='blue', s=200, marker='*', label='Top 3 Ativos', zorder=5)

    ax.set_xlabel('Taxa Máxima (% CDI)')
    ax.set_ylabel('ROA Estimada Anual (%)')
    ax.set_title('Mapa de Oportunidades em Renda Fixa', fontsize=16)
    fig.colorbar(scatter, ax=ax, label='Prazo de Carência (dias)')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    
    return fig, top3

# --- Interface do Aplicativo ---
st.title('Analisador Interativo de Renda Fixa 📊')
st.write("Esta ferramenta ajuda a visualizar as melhores oportunidades em ativos de renda fixa, cruzando a **Taxa Máxima** com o **ROA (Retorno sobre Ativos)** da instituição.")

st.sidebar.header('1. Carregue seus dados')
uploaded_file = st.sidebar.file_uploader("Escolha o arquivo Excel", type="xlsx")

if uploaded_file is None:
    st.info("Por favor, carregue um arquivo Excel para começar.")
    st.stop()

df_original = load_and_process_data(uploaded_file)
df = df_original.copy()

st.sidebar.header('2. Filtre as Oportunidades')
with st.sidebar.expander("ℹ️ Como Usar o Aplicativo"):
    st.markdown(
        """
        1. **Carregue seus dados:** Use o botão acima.
        2. **Filtre as oportunidades:** Use os controles abaixo para refinar sua busca.
        3. **Analise os resultados:** O gráfico mostra a relação Risco vs. Retorno. A cor indica o prazo e os Top 3 são destacados com uma ⭐.
        """
    )

if 'Prazo Carência (dias)' in df.columns:
    min_prazo = df['Prazo Carência (dias)'].min()
    max_prazo = df['Prazo Carência (dias)'].max()
    selected_prazo = st.sidebar.slider(
        'Prazo de Carência (dias)',
        min_value=min_prazo, max_value=max_prazo, value=(min_prazo, max_prazo)
    )
    df_filtered = df[df['Prazo Carência (dias)'].between(selected_prazo[0], selected_prazo[1])]
else:
    df_filtered = df.copy()

if 'Indexador' in df_filtered.columns:
    indexadores_disponiveis = df_filtered['Indexador'].unique()
    selected_indexador = st.sidebar.multiselect(
        'Indexador do Ativo',
        options=indexadores_disponiveis, default=indexadores_disponiveis
    )
    df_filtered = df_filtered[df_filtered['Indexador'].isin(selected_indexador)]

st.header('Resultados Filtrados')
if df_filtered.empty:
    st.warning("Nenhum ativo encontrado com os filtros selecionados.")
else:
    fig, top3 = create_plot(df_filtered)
    
    if fig:
        st.pyplot(fig)

        ## CORREÇÃO 3: TABELAS - Lógica que previne o KeyError
        cols_to_display = ['Ativo', 'Tax.Máx', 'ROA E. Aprox.', 'Prazo Carência (dias)', 'Vencimento', 'Indexador', 'Emissor']
        
        available_cols_top3 = [col for col in cols_to_display if col in top3.columns]
        available_cols_all = [col for col in cols_to_display if col in df_filtered.columns]

        if not top3.empty and available_cols_top3:
            st.subheader('⭐ Top 3 Oportunidades Filtradas')
            st.dataframe(top3[available_cols_top3])

        if available_cols_all:
            st.subheader('Todos os Dados Filtrados')
            st.dataframe(df_filtered[available_cols_all])
    else:
        st.warning("Não foi possível gerar o gráfico com os dados filtrados.")