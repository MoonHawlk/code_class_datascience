import os; os.system("python -m pip install streamlit pandas numpy matplotlib seaborn")

# =========================
# 1. Imports e Configuração
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import calendar

st.set_page_config(
    page_title="Análise de Consumo de Cerveja",
    layout="wide"
)

sns.set_style("whitegrid")

# =========================
# 2. Carga e Tratamento dos Dados
# =========================
@st.cache_data
def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    temp_columns = [
        'Temperatura Media (C)',
        'Temperatura Minima (C)',
        'Temperatura Maxima (C)',
        'Precipitacao (mm)'
    ]

    for col in temp_columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(',', '.', regex=False)
            .astype(float)
        )

    df = df.dropna(subset=['Data'])
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')

    day_name_mapping = {
        'Monday': 'Segunda-feira',
        'Tuesday': 'Terça-feira',
        'Wednesday': 'Quarta-feira',
        'Thursday': 'Quinta-feira',
        'Friday': 'Sexta-feira',
        'Saturday': 'Sábado',
        'Sunday': 'Domingo'
    }

    df['Dia da Semana'] = df['Data'].dt.day_name().map(day_name_mapping)
    df['Mes'] = df['Data'].dt.month

    return df


df = load_and_prepare_data("data/Consumo_cerveja.csv")

# =========================
# 3. Apresentação
# =========================
st.title("🍺 Análise de Consumo de Cerveja")
st.markdown("### Entendendo os Padrões de Consumo através de Dados")

st.markdown("""
**Grupo:**
- Ana Cristina Oliveira Silva

- Filipe Vasconcelos Moreno

- João Henrique Lampropulos Rietra

- Rennan Pontes Cardoso
""")

st.divider()

# =========================
# 4. VISÃO GERAL - Overview dos Dados
# =========================
st.header("📊 Visão Geral dos Dados")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total de Dias Analisados",
        len(df),
        help="Período completo de coleta de dados"
    )

with col2:
    st.metric(
        "Consumo Médio Diário",
        f"{df['Consumo de cerveja (litros)'].mean():.1f}L",
        help="Média de consumo por dia"
    )

with col3:
    st.metric(
        "Temperatura Média",
        f"{df['Temperatura Media (C)'].mean():.1f}°C",
        help="Temperatura média do período"
    )

with col4:
    correlacao_temp = df['Temperatura Media (C)'].corr(df['Consumo de cerveja (litros)'])
    st.metric(
        "Correlação Temp×Consumo",
        f"{correlacao_temp:.3f}",
        help="Quanto mais próximo de 1, maior a relação positiva"
    )

st.divider()

# =========================
# 5. PADRÃO TEMPORAL - Consumo ao Longo do Tempo
# =========================
st.header("📈 Padrão Temporal de Consumo")
st.markdown("**Como o consumo de cerveja varia ao longo do ano?**")

fig_ts, ax = plt.subplots(figsize=(20, 8))

# Linha principal
sns.lineplot(
    x='Data',
    y='Consumo de cerveja (litros)',
    data=df,
    label='Consumo Diário',
    alpha=0.7,
    linewidth=1.5,
    color='steelblue',
    ax=ax
)

# Destaque finais de semana
weekend_data = df[df['Final de Semana'] == True]
sns.scatterplot(
    x='Data',
    y='Consumo de cerveja (litros)',
    data=weekend_data,
    s=50,
    color='orange',
    label='Finais de Semana',
    alpha=0.8,
    ax=ax
)

# Destaque segundas-feiras
monday_data = df[df['Dia da Semana'] == 'Segunda-feira']
sns.scatterplot(
    x='Data',
    y='Consumo de cerveja (litros)',
    data=monday_data,
    marker='X',
    s=60,
    color='red',
    label='Segundas-feiras',
    alpha=0.7,
    ax=ax
)

ax.set_title("Consumo Diário de Cerveja ao Longo do Tempo", fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel("Período", fontsize=13)
ax.set_ylabel("Consumo (Litros)", fontsize=13)

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%Y'))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)

fig_ts.tight_layout()
st.pyplot(fig_ts)

st.info("💡 **Insight**: Observe os picos de consumo nos finais de semana (pontos laranjas) e a queda nas segundas-feiras (marcadores vermelhos).")

st.divider()

# =========================
# 6. RANKING POR DIA DA SEMANA
# =========================
st.header("🏆 Ranking: Consumo por Dia da Semana")
st.markdown("**Qual dia da semana tem o maior consumo médio?**")

col_ranking, col_grafico = st.columns([1, 2])

with col_ranking:
    tabela_media = (
        df.groupby('Dia da Semana', as_index=False)
          .agg(Media_Consumo_Litros=('Consumo de cerveja (litros)', 'mean'))
          .sort_values(by='Media_Consumo_Litros', ascending=False)
    )
    tabela_media['Media_Consumo_Litros'] = tabela_media['Media_Consumo_Litros'].round(2)
    tabela_media.index = range(1, len(tabela_media) + 1)
    
    st.dataframe(
        tabela_media,
        use_container_width=True,
        height=280
    )

with col_grafico:
    fig_rank, ax = plt.subplots(figsize=(10, 6))
    
    cores = ['#FFD700' if i == 0 else '#C0C0C0' if i == 1 else '#CD7F32' if i == 2 else 'steelblue' 
             for i in range(len(tabela_media))]
    
    bars = ax.barh(
        tabela_media['Dia da Semana'],
        tabela_media['Media_Consumo_Litros'],
        color=cores,
        edgecolor='black',
        linewidth=1.5
    )
    
    # Adicionar valores
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(
            width + 0.5,
            bar.get_y() + bar.get_height()/2,
            f'{width:.1f}L',
            ha='left',
            va='center',
            fontsize=11,
            fontweight='bold'
        )
    
    ax.set_xlabel('Consumo Médio (Litros)', fontsize=12, fontweight='bold')
    ax.set_title('Consumo Médio por Dia da Semana', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()
    
    fig_rank.tight_layout()
    st.pyplot(fig_rank)

st.success("🥇 **Destaque**: Sábado é o campeão absoluto de consumo, seguido por Sexta-feira e Domingo!")

st.divider()

# =========================
# 7. ANÁLISE MENSAL DETALHADA
# =========================
st.header("📅 Análise Detalhada por Mês")
st.markdown("**Como os padrões semanais variam ao longo dos meses?**")

ordem_dias = [
    'Segunda-feira', 'Terça-feira', 'Quarta-feira',
    'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo'
]

fig, axes = plt.subplots(3, 4, figsize=(22, 13))
axes = axes.flatten()

for mes in range(1, 13):
    ax = axes[mes - 1]
    df_mes = df[df['Mes'] == mes]
    
    consumo_medio_mes = (
        df_mes
        .groupby('Dia da Semana')['Consumo de cerveja (litros)']
        .mean()
        .reindex(ordem_dias)
        .dropna()
    )
    
    bars = ax.bar(
        consumo_medio_mes.index,
        consumo_medio_mes.values,
        color='steelblue',
        edgecolor='white',
        linewidth=0.7,
        zorder=2
    )
    
    if len(consumo_medio_mes) > 0:
        idx_max = consumo_medio_mes.values.argmax()
        idx_min = consumo_medio_mes.values.argmin()
        
        max_bar = bars[idx_max]
        min_bar = bars[idx_min]
        
        # Retângulo verde para maior
        rect_max = Rectangle(
            (max_bar.get_x() - 0.05, 0),
            max_bar.get_width() + 0.1,
            max_bar.get_height(),
            fill=False,
            edgecolor='green',
            linewidth=3.5,
            zorder=10
        )
        ax.add_patch(rect_max)
        
        # Retângulo vermelho para menor
        rect_min = Rectangle(
            (min_bar.get_x() - 0.05, 0),
            min_bar.get_width() + 0.1,
            min_bar.get_height(),
            fill=False,
            edgecolor='red',
            linewidth=3.5,
            zorder=10
        )
        ax.add_patch(rect_min)
        
        ax.set_ylim(0, consumo_medio_mes.max() * 1.15)
    
    ax.set_title(calendar.month_name[mes], fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=1)

for ax in axes:
    if not ax.has_data():
        ax.axis('off')

fig.suptitle(
    "Consumo Médio de Cerveja por Dia da Semana em Cada Mês\n"
    "🟢 Maior consumo do mês  |  🔴 Menor consumo do mês",
    fontsize=17,
    fontweight='bold',
    y=0.995
)

fig.tight_layout(rect=[0, 0, 1, 0.97])
st.pyplot(fig)

st.info("💡 **Padrão Identificado**: Em todos os meses, os finais de semana (especialmente sábado) mantêm o maior consumo.")

st.divider()

# =========================
# 8. RELAÇÃO CLIMA × CONSUMO
# =========================
st.header("🌡️ Impacto do Clima no Consumo")
st.markdown("### Temperatura e Precipitação: Como Influenciam?")

# Subcabeçalho: Temperatura
st.subheader("1️⃣ Relação com a Temperatura")

fig_temp, ax = plt.subplots(figsize=(14, 6))

scatter_temp = ax.scatter(
    df['Temperatura Media (C)'],
    df['Consumo de cerveja (litros)'],
    alpha=0.6,
    s=50,
    c=df['Temperatura Media (C)'],
    cmap='RdYlBu_r',
    edgecolors='black',
    linewidth=0.3
)

# Linha de tendência
z_temp = np.polyfit(df['Temperatura Media (C)'], df['Consumo de cerveja (litros)'], 1)
p_temp = np.poly1d(z_temp)
ax.plot(
    df['Temperatura Media (C)'].sort_values(),
    p_temp(df['Temperatura Media (C)'].sort_values()),
    "r--",
    linewidth=3,
    label=f'Tendência: y = {z_temp[0]:.2f}x + {z_temp[1]:.2f}'
)

cbar = plt.colorbar(scatter_temp, ax=ax)
cbar.set_label('Temperatura (°C)', rotation=270, labelpad=20, fontsize=11)

ax.set_xlabel('Temperatura Média (°C)', fontsize=13, fontweight='bold')
ax.set_ylabel('Consumo de Cerveja (litros)', fontsize=13, fontweight='bold')
ax.set_title('Relação Positiva: Quanto Mais Quente, Maior o Consumo', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

st.pyplot(fig_temp)

corr_temp = df['Temperatura Media (C)'].corr(df['Consumo de cerveja (litros)'])
st.metric("📊 Correlação Temperatura × Consumo", f"{corr_temp:.3f}", help="Correlação positiva forte!")

st.divider()

# Subcabeçalho: Precipitação
st.subheader("2️⃣ Relação com a Precipitação (Chuva)")

fig_prec, ax = plt.subplots(figsize=(14, 6))

scatter = ax.scatter(
    df['Precipitacao (mm)'],
    df['Consumo de cerveja (litros)'],
    alpha=0.6,
    s=50,
    c=df['Temperatura Media (C)'],
    cmap='YlOrRd',
    edgecolors='black',
    linewidth=0.3
)

z = np.polyfit(df['Precipitacao (mm)'], df['Consumo de cerveja (litros)'], 1)
p = np.poly1d(z)
ax.plot(
    df['Precipitacao (mm)'].sort_values(),
    p(df['Precipitacao (mm)'].sort_values()),
    "r--",
    linewidth=3,
    label=f'Tendência: y = {z[0]:.2f}x + {z[1]:.2f}'
)

cbar2 = plt.colorbar(scatter, ax=ax)
cbar2.set_label('Temperatura Média (°C)', rotation=270, labelpad=20, fontsize=11)

ax.set_xlabel('Precipitação (mm)', fontsize=13, fontweight='bold')
ax.set_ylabel('Consumo de Cerveja (litros)', fontsize=13, fontweight='bold')
ax.set_title('Relação Inversa: Quanto Mais Chove, Menor o Consumo', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

st.pyplot(fig_prec)

correlacao_prec = df['Precipitacao (mm)'].corr(df['Consumo de cerveja (litros)'])
st.metric("📊 Correlação Precipitação × Consumo", f"{correlacao_prec:.3f}", help="Correlação negativa - são inversamente proporcionais!")

st.divider()

# =========================
# 9. CONSUMO POR TIPO DE CHUVA
# =========================
st.header("☔ Análise por Intensidade de Chuva")
st.markdown("**Como diferentes níveis de precipitação afetam o consumo?**")

# Categorizar precipitação
df['Tipo_Chuva'] = pd.cut(
    df['Precipitacao (mm)'],
    bins=[-0.1, 0.1, 5, 15, 50, 200],
    labels=['Sem chuva', 'Chuva leve', 'Chuva moderada', 'Chuva forte', 'Chuva intensa']
)

consumo_por_tipo = (
    df.groupby('Tipo_Chuva', observed=True)['Consumo de cerveja (litros)']
    .agg(['mean', 'std', 'count'])
    .reset_index()
)

fig, ax = plt.subplots(figsize=(14, 7))

cores = ['#27ae60', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c']

bars = ax.bar(
    consumo_por_tipo['Tipo_Chuva'],
    consumo_por_tipo['mean'],
    yerr=consumo_por_tipo['std'],
    capsize=10,
    color=cores[:len(consumo_por_tipo)],
    edgecolor='black',
    linewidth=2,
    alpha=0.85
)

for i, bar in enumerate(bars):
    height = bar.get_height()
    std = consumo_por_tipo.iloc[i]['std']
    count = int(consumo_por_tipo.iloc[i]['count'])
    
    ax.text(
        bar.get_x() + bar.get_width()/2.,
        height + std + 1,
        f'{height:.1f}L',
        ha='center',
        va='bottom',
        fontsize=13,
        fontweight='bold'
    )
    
    ax.text(
        bar.get_x() + bar.get_width()/2.,
        8,
        f'{count} dias',
        ha='center',
        va='bottom',
        fontsize=10,
        color='white',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8)
    )

idx_max = consumo_por_tipo['mean'].idxmax()
idx_min = consumo_por_tipo['mean'].idxmin()

rect_max = Rectangle(
    (bars[idx_max].get_x() - 0.08, 0),
    bars[idx_max].get_width() + 0.16,
    bars[idx_max].get_height(),
    fill=False,
    edgecolor='green',
    linewidth=5,
    zorder=10,
    label='✅ Maior consumo'
)
ax.add_patch(rect_max)

rect_min = Rectangle(
    (bars[idx_min].get_x() - 0.08, 0),
    bars[idx_min].get_width() + 0.16,
    bars[idx_min].get_height(),
    fill=False,
    edgecolor='red',
    linewidth=5,
    zorder=10,
    label='❌ Menor consumo'
)
ax.add_patch(rect_min)

ax.set_ylabel('Consumo Médio de Cerveja (litros)', fontsize=14, fontweight='bold')
ax.set_xlabel('Categoria de Precipitação', fontsize=14, fontweight='bold')
ax.set_title(
    'Impacto da Precipitação no Consumo de Cerveja\n'
    'Relação Inversamente Proporcional Claramente Visível',
    fontsize=16,
    fontweight='bold',
    pad=20
)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
st.pyplot(fig)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📉 Correlação Chuva×Consumo", f"{correlacao_prec:.3f}")
with col2:
    st.metric("🏆 Maior Consumo", f"{consumo_por_tipo.iloc[idx_max]['Tipo_Chuva']}")
with col3:
    st.metric("📉 Menor Consumo", f"{consumo_por_tipo.iloc[idx_min]['Tipo_Chuva']}")

st.success("✅ **Conclusão**: Dias sem chuva apresentam consumo significativamente maior que dias chuvosos!")

st.divider()

# =========================
# 10. EVOLUÇÃO TEMPORAL DUAL-AXIS
# =========================
st.header("📊 Evolução Temporal Comparativa")
st.markdown("**Visualizando a inversão: Chuva ↓ Consumo ↑**")

df_agrupado = df.groupby(df['Data'].dt.to_period('W')).agg({
    'Precipitacao (mm)': 'sum',
    'Consumo de cerveja (litros)': 'mean'
}).reset_index()
df_agrupado['Data'] = df_agrupado['Data'].dt.to_timestamp()

fig, ax1 = plt.subplots(figsize=(16, 7))

color1 = '#2E86AB'
ax1.set_xlabel('Período (Semanas)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Consumo de Cerveja (litros)', color=color1, fontsize=13, fontweight='bold')
line1 = ax1.plot(
    df_agrupado['Data'],
    df_agrupado['Consumo de cerveja (litros)'],
    color=color1,
    linewidth=3,
    label='Consumo de Cerveja',
    marker='o',
    markersize=4
)
ax1.tick_params(axis='y', labelcolor=color1, labelsize=11)
ax1.fill_between(
    df_agrupado['Data'],
    df_agrupado['Consumo de cerveja (litros)'],
    alpha=0.3,
    color=color1
)

ax2 = ax1.twinx()
color2 = '#A23B72'
ax2.set_ylabel('Precipitação Semanal (mm)', color=color2, fontsize=13, fontweight='bold')
line2 = ax2.plot(
    df_agrupado['Data'],
    df_agrupado['Precipitacao (mm)'],
    color=color2,
    linewidth=3,
    label='Precipitação',
    linestyle='--',
    marker='s',
    markersize=4
)
ax2.tick_params(axis='y', labelcolor=color2, labelsize=11)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=12, framealpha=0.9)

ax1.set_title(
    'Padrão Inverso: Quando Chove Mais, Consome-se Menos Cerveja\n'
    'Análise Semanal ao Longo do Período',
    fontsize=16,
    fontweight='bold',
    pad=20
)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

fig.tight_layout()
st.pyplot(fig)

st.info("💡 **Observação**: Note como os picos de chuva (linha roxa tracejada) coincidem com quedas no consumo (área azul).")

st.divider()

# =========================
# 11. MATRIZ DE CORRELAÇÃO FINAL
# =========================
st.header("🔗 Matriz de Correlação Completa")
st.markdown("**Visão geral das relações entre todas as variáveis**")

fig_corr, ax = plt.subplots(figsize=(12, 9))

correlation_matrix = df.select_dtypes(include=['float64', 'int64', 'bool']).corr()

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='RdBu_r',
    fmt=".3f",
    ax=ax,
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={"shrink": 0.8, "label": "Coeficiente de Correlação"},
    annot_kws={"size": 10, "weight": "bold"}
)

ax.set_title("Matriz de Correlação - Todas as Variáveis", fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
st.pyplot(fig_corr)

st.markdown("""
**🔍 Como interpretar:**
- **Valores próximos a +1**: Correlação positiva forte (quando uma sobe, a outra também sobe)
- **Valores próximos a -1**: Correlação negativa forte (quando uma sobe, a outra desce)
- **Valores próximos a 0**: Pouca ou nenhuma correlação linear
""")

st.divider()

# =========================
# 12. CONCLUSÕES FINAIS
# =========================
st.header("📝 Conclusões Principais")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### ✅ Fatores que **AUMENTAM** o consumo:
    
    1. **Finais de semana** (especialmente sábados)
    2. **Temperaturas elevadas**
    3. **Ausência de chuva**
    4. **Sextas-feiras** (preparação para o fim de semana)
    """)

with col2:
    st.markdown("""
    ### ❌ Fatores que **REDUZEM** o consumo:
    
    1. **Dias úteis** (segunda a quinta-feira)
    2. **Precipitação intensa**
    3. **Temperaturas mais baixas**
    4. **Segundas-feiras** (menor consumo da semana)
    """)

st.success("""
### 🎯 **Insight Principal para Negócios**

O consumo de cerveja é **fortemente influenciado** por:
- **Padrões sociais** (fim de semana vs. dias úteis)
- **Condições climáticas** (temperatura e precipitação)

**Recomendação**: Estoques devem ser ajustados considerando previsões meteorológicas e calendário, 
maximizando disponibilidade em períodos de alta temperatura e finais de semana sem previsão de chuva.
""")

st.divider()