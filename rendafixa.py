import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- Configurações da Página e Funções ---

# Configura o layout da página para ser mais largo
st.set_page_config(layout="wide")

# Cache para evitar recarregar e reprocessar os dados a cada interação
@st.cache_data
def load_and_process_data(uploaded_file):
    """Carrega o arquivo excel e aplica a limpeza e engenharia de features iniciais."""
    df = pd.read_excel(uploaded_file, sheet_name='Resultado')

    # 1. Limpeza das colunas
    def clean_columns(df):
        # Converte 'Tax.Máx' para float
        df['Tax.Máx'] = (
            df['Tax.Máx']
            .astype(str)
            .str.replace('% CDI', '', regex=False)
            .str.replace(',', '.')
            .str.extract(r'(\d+\.?\d*)')[0]
            .astype(float)
        )
        # Converte 'ROA E. Aprox.' para float
        df['ROA E. Aprox.'] = (
            df['ROA E. Aprox.']
            .astype(str)
            .str.replace('%', '', regex=False)
            .str.replace(',', '.')
            .str.extract(r'(\d+\.?\d*)')[0]
            .astype(float)
        )
        return df.dropna(subset=['Tax.Máx', 'ROA E. Aprox.', 'Vencimento'])

    df = clean_columns(df)

    # 2. Calcular prazo de carência em dias
    def calcular_prazo_carencia(row):
        hoje = datetime.now().date()
        try:
            venc = pd.to_datetime(row['Vencimento']).date()
        except:
            return np.nan # Retorna NaN se a data de vencimento for inválida

        carencia = str(row.get('Carência', '')).strip().lower()

        if carencia in ['nan', '', 'vencimento', 'no vencimento']:
            prazo = (venc - hoje).days
        else:
            try:
                # Tenta converter para número (ex: '90 dias')
                prazo_num = int(''.join(filter(str.isdigit, carencia)))
                prazo = prazo_num
            except:
                prazo = (venc - hoje).days # Fallback para o vencimento

        return max(prazo, 0)

    df['Prazo Carência (dias)'] = df.apply(calcular_prazo_carencia, axis=1)
    df.dropna(subset=['Prazo Carência (dias)'], inplace=True) # Remove linhas onde o prazo não pôde ser calculado
    df['Prazo Carência (dias)'] = df['Prazo Carência (dias)'].astype(int)

    return df

def create_plot(df_filtered):
    """Cria o gráfico de dispersão com base no dataframe filtrado."""
    if df_filtered.empty:
        return None

    # Recalcula o Score apenas para os dados filtrados
    tax_min, tax_max = df_filtered['Tax.Máx'].min(), df_filtered['Tax.Máx'].max()
    roa_min, roa_max = df_filtered['ROA E. Aprox.'].min(), df_filtered['ROA E. Aprox.'].max()
    
    # Evita divisão por zero se houver apenas um ponto de dado
    if (tax_max - tax_min) > 0:
        df_filtered['Taxa_norm'] = (df_filtered['Tax.Máx'] - tax_min) / (tax_max - tax_min)
    else:
        df_filtered['Taxa_norm'] = 0.5
        
    if (roa_max - roa_min) > 0:
        df_filtered['ROA_norm'] = (df_filtered['ROA E. Aprox.'] - roa_min) / (roa_max - roa_min)
    else:
        df_filtered['ROA_norm'] = 0.5
        
    df_filtered['Score'] = df_filtered['Taxa_norm'] + df_filtered['ROA_norm']

    # Inicia a figura do Matplotlib
    fig, ax = plt.subplots(figsize=(14, 8))

    # Mapa de calor de densidade
    sns.kdeplot(
        x=df_filtered['Tax.Máx'], y=df_filtered['ROA E. Aprox.'],
        cmap="YlOrRd", fill=True, thresh=0.05, levels=100, alpha=0.3, ax=ax
    )

    # Scatter plot colorido pelo prazo de carência
    norm = plt.Normalize(df_filtered['Prazo Carência (dias)'].min(), df_filtered['Prazo Carência (dias)'].max())
    scatter = ax.scatter(
        df_filtered['Tax.Máx'], df_filtered['ROA E. Aprox.'],
        c=df_filtered['Prazo Carência (dias)'], cmap='RdYlGn_r', s=90, edgecolor='k', alpha=0.85, norm=norm
    )

    # Destacar os 3 melhores pelo score combinado
    top3 = df_filtered.sort_values('Score', ascending=False).head(3)
    for _, row in top3.iterrows():
        ax.text(row['Tax.Máx'], row['ROA E. Aprox.'], f"  {row['Ativo']}", fontsize=10, weight='bold', color='navy', va='center')
        ax.scatter(row['Tax.Máx'], row['ROA E. Aprox.'], c='blue', s=200, marker='*', label='Top 3 Ativos', zorder=5)


    ax.set_xlabel('Taxa Máxima (% CDI)')
    ax.set_ylabel('ROA Estimada Anual (%)')
    ax.set_title('Mapa de Oportunidades em Renda Fixa - Relação Risco vs. Retorno', fontsize=16)
    fig.colorbar(scatter, ax=ax, label='Prazo de Carência (dias)')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Evita duplicar labels na legenda
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    
    return fig, top3

# --- Interface do Aplicativo ---

st.title('Analisador Interativo de Renda Fixa 📊')

st.write("""
Esta ferramenta ajuda a visualizar as melhores oportunidades em ativos de renda fixa de emissão bancária,
cruzando a **Taxa Máxima** com o **ROA (Retorno sobre Ativos)** da instituição.
""")

# 1. Upload do arquivo na barra lateral
st.sidebar.header('1. Carregue seus dados')
uploaded_file = st.sidebar.file_uploader(
    "Escolha o arquivo 'produtos-renda-fixa.xlsx'", type="xlsx"
)

if uploaded_file is None:
    st.info("Por favor, carregue um arquivo Excel para começar.")
    st.stop()

# Carrega e processa os dados (usando cache)
df_original = load_and_process_data(uploaded_file)
df = df_original.copy() # Cria uma cópia para manipulação

# 2. Filtros interativos na barra lateral
st.sidebar.header('2. Filtre as Oportunidades')

# Filtro por Prazo de Carência
min_prazo = df['Prazo Carência (dias)'].min()
max_prazo = df['Prazo Carência (dias)'].max()
selected_prazo = st.sidebar.slider(
    'Prazo de Carência (dias)',
    min_value=min_prazo,
    max_value=max_prazo,
    value=(min_prazo, max_prazo) # Define o valor inicial como o intervalo completo
)

# Filtro por Indexador
indexadores_disponiveis = df['Indexador'].unique()
selected_indexador = st.sidebar.multiselect(
    'Indexador do Ativo',
    options=indexadores_disponiveis,
    default=indexadores_disponiveis # Deixa todos selecionados por padrão
)

# Aplica os filtros ao dataframe
df_filtered = df[
    (df['Prazo Carência (dias)'].between(selected_prazo[0], selected_prazo[1])) &
    (df['Indexador'].isin(selected_indexador))
]

# --- Exibição dos Resultados ---

st.header('Resultados Filtrados')

if df_filtered.empty:
    st.warning("Nenhum ativo encontrado com os filtros selecionados. Tente ajustar os filtros.")
else:
    # Cria e exibe o gráfico
    fig, top3 = create_plot(df_filtered)
    st.pyplot(fig)

    # Exibe a tabela com os 3 melhores
    st.subheader('⭐ Top 3 Oportunidades Filtradas')
    st.dataframe(top3[['Ativo', 'Tax.Máx', 'ROA E. Aprox.', 'Prazo Carência (dias)', 'Vencimento', 'Indexador', 'Emissor']])

    # Exibe a tabela de dados completa
    st.subheader('Todos os Dados Filtrados')
    st.dataframe(df_filtered[['Ativo', 'Tax.Máx', 'ROA E. Aprox.', 'Prazo Carência (dias)', 'Vencimento', 'Indexador', 'Emissor']])import streamlit as st
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
        # Garante que a conversão só acontece se a coluna existir
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
        
        # Garante que só vai dropar NA se as colunas existirem
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

    # 1. CORREÇÃO: Tamanho do gráfico ajustado de (14, 8) para (10, 6)
    fig, ax = plt.subplots(figsize=(10, 6))
    
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
        for _, row in top3.iterrows():
            # 2. CORREÇÃO: Tamanho da fonte do texto reduzido para melhor legibilidade
            ax.text(row['Tax.Máx'], row['ROA E. Aprox.'], f"  {row['Ativo']}", fontsize=9, weight='bold', color='navy', va='center')
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
    
    # 3. CORREÇÃO: Garante que o layout se ajuste bem
    plt.tight_layout()
    
    return fig, top3

# --- Interface do Aplicativo ---

st.title('Analisador Interativo de Renda Fixa 📊')
st.write("""
Esta ferramenta ajuda a visualizar as melhores oportunidades em ativos de renda fixa,
cruzando a **Taxa Máxima** com o **ROA (Retorno sobre Ativos)** da instituição.
""")

st.sidebar.header('1. Carregue seus dados')
uploaded_file = st.sidebar.file_uploader(
    "Escolha o arquivo Excel", type="xlsx"
)

if uploaded_file is None:
    st.info("Por favor, carregue um arquivo Excel para começar.")
    st.stop()

df_original = load_and_process_data(uploaded_file)
df = df_original.copy()

st.sidebar.header('2. Filtre as Oportunidades')
with st.sidebar.expander("ℹ️ Como Usar o Aplicativo"):
    st.markdown(
        """
        1.  **Carregue seus dados:** Use o botão acima para carregar sua planilha.
        2.  **Filtre as oportunidades:** Use os controles abaixo para refinar sua busca por prazo e indexador.
        3.  **Analise os resultados:**
            * O **gráfico** mostra a relação Risco vs. Retorno.
            * A **cor** indica o prazo de carência.
            * Os **Top 3** ativos são destacados com uma ⭐.
            * As **tabelas** na tela principal mostram os detalhes.
        """
    )

if 'Prazo Carência (dias)' in df.columns:
    min_prazo = df['Prazo Carência (dias)'].min()
    max_prazo = df['Prazo Carência (dias)'].max()
    selected_prazo = st.sidebar.slider(
        'Prazo de Carência (dias)',
        min_value=min_prazo, max_value=max_prazo,
        value=(min_prazo, max_prazo)
    )
    df_filtered = df[df['Prazo Carência (dias)'].between(selected_prazo[0], selected_prazo[1])]
else:
    df_filtered = df.copy()

if 'Indexador' in df.columns:
    indexadores_disponiveis = df_filtered['Indexador'].unique()
    selected_indexador = st.sidebar.multiselect(
        'Indexador do Ativo',
        options=indexadores_disponiveis,
        default=indexadores_disponiveis
    )
    df_filtered = df_filtered[df_filtered['Indexador'].isin(selected_indexador)]

st.header('Resultados Filtrados')
if df_filtered.empty:
    st.warning("Nenhum ativo encontrado com os filtros selecionados.")
else:
    fig, top3 = create_plot(df_filtered)
    
    if fig:
        st.pyplot(fig)

        # 4. CORREÇÃO: Lógica defensiva para evitar o KeyError
        # Define a lista de colunas que idealmente queremos mostrar
        cols_to_display = [
            'Ativo', 'Tax.Máx', 'ROA E. Aprox.', 'Prazo Carência (dias)', 
            'Vencimento', 'Indexador', 'Emissor'
        ]
        
        # Filtra a lista para incluir apenas as colunas que REALMENTE existem no DataFrame
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