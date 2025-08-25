import random

import joblib
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from convert import preprocess_creditcard_data

model_path = './model/random_forest_model.joblib'
clf = joblib.load(model_path)


# -----------------------------
# 页面配置
# -----------------------------
st.set_page_config(page_title="信用卡欺诈分析仪表盘", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: #000000;
            color: white;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            background-color: #000000;
        }
        .stMetric, .stDataFrame, .stChart {
            background-color: transparent;
            border-radius: 0.5rem;
            padding: 1rem;
        }
        /* 设置标题颜色为白色 */
        h1 {
            color: white;
        }
        button[kind="header"] {
            display: block !important;
        }
    </style>
""", unsafe_allow_html=True)

# 左侧菜单栏
with st.sidebar:
    st.title("菜单")
    uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])

if uploaded_file is not None:
    # 加载数据
    df = pd.read_csv(uploaded_file, index_col='Unnamed: 0')
    # 数据预处理
    df1 = df.copy()
    X_test, _ = preprocess_creditcard_data(df1)
    # 模型预测
    predictions = clf.predict(X_test)

    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['year'] = df['trans_date_trans_time'].dt.year
    df['month'] = df['trans_date_trans_time'].dt.month
    df['hour'] = df['trans_date_trans_time'].dt.hour
    # 将预测结果添加到 DataFrame 中
    df['is_fraud'] = predictions
else:
    # 没有上传文件时，初始化数据
    df = pd.DataFrame({
        'trans_date_trans_time': [],
        'cc_num': [],
        'merchant': [],
        'category': [],
        'amt': [],
        'first': [],
        'last': [],
        'gender': [],
        'street': [],
        'city': [],
        'state': [],
        'zip': [],
        'lat': [],
        'long': [],
        'city_pop': [],
        'job': [],
        'dob': [],
        'trans_num': [],
        'unix_time': [],
        'merch_lat': [],
        'merch_long': [],
        'is_fraud': []
    })
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['year'] = df['trans_date_trans_time'].dt.year
    df['month'] = df['trans_date_trans_time'].dt.month
    df['hour'] = df['trans_date_trans_time'].dt.hour

# -----------------------------
# 计算指标
# -----------------------------
total_transactions = len(df)
fraud_transactions = df['is_fraud'].sum()
fraud_ratio = fraud_transactions / total_transactions if total_transactions > 0 else 0

# 修改标题样式，确保颜色为白色
st.markdown("<h3 style='color: white;'></h3>", unsafe_allow_html=True)

st.markdown("<h3 style='color: white; text-align: center;'>💳 信用卡欺诈检测仪表盘</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])

# 使用 HTML 和 CSS 模拟 st.metric 并指定颜色
col1.markdown(f"""
    <div style="text-align: center; color: white;">
        <p style="font-size: 14px;">交易总数</p>
        <p style="font-size: 24px; color: #00FF7F;">{total_transactions:,}</p>
    </div>
""", unsafe_allow_html=True)

col2.markdown(f"""
    <div style="text-align: center; color: white;">
        <p style="font-size: 14px;">欺诈交易数</p>
        <p style="font-size: 24px; color: #FF4C4C;">{fraud_transactions:,}</p>
    </div>
""", unsafe_allow_html=True)

col3.markdown(f"""
    <div style="text-align: center; color: white;">
        <p style="font-size: 14px;">欺诈率</p>
        <p style="font-size: 24px; color: #FFFF00;">{fraud_ratio:.2%}</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# 准备图表数据
# -----------------------------
if df.empty:
    fraud_pie = pd.DataFrame({
        'is_fraud': [0, 1],
        'count': [0, 1],
        'label': ['正常交易', '欺诈交易']
    })
else:
    fraud_pie = df['is_fraud'].value_counts().reset_index()
    fraud_pie.columns = ['is_fraud', 'count']
    fraud_pie['label'] = fraud_pie['is_fraud'].map({0: '正常交易', 1: '欺诈交易'})

fraud_by_cat = df[df['is_fraud'] == 1]['category'].value_counts().reset_index()
fraud_by_cat.columns = ['商户类别', '欺诈数量']

fraud_hour = df[df['is_fraud'] == 1]['hour'].value_counts().sort_index().reset_index()
fraud_hour.columns = ['小时', '欺诈数量']

state_fraud = df[df['is_fraud'] == 1]['state'].value_counts().reset_index()
state_fraud.columns = ['州', '欺诈数量']

# 计算年龄分布
df['age'] = df['dob'].apply(lambda x: 2023 - int(x.split('-')[0]) if isinstance(x, str) else 0)
fraud_age = df[df['is_fraud'] == 1]['age'].dropna()

# 月份分布
fraud_month = df[df['is_fraud'] == 1]['month'].value_counts().sort_index().reset_index()
fraud_month.columns = ['月份', '欺诈数量']

# -----------------------------
# 布局：地图居中 + 两边图表
# -----------------------------
left_col, center_col, right_col = st.columns([1.3, 2.2, 1.3])
# 左 - 饼图
with left_col:
    fig_pie = px.pie(
        fraud_pie, names='label', values='count',
        title='欺诈与正常交易比例',
        hole=0.4,
        color_discrete_sequence=['#FFFF00', '#800080']  # 黄色与紫色
    )
    fig_pie.update_layout(
        height=320,
        template='plotly_dark',
        margin=dict(t=40, b=0, l=0, r=0),
        font=dict(family="Microsoft YaHei", color="white", size=14),
        paper_bgcolor='rgba(0,0,0,0)',  # 透明背景
        plot_bgcolor='rgba(0,0,0,0)',
        title_font=dict(color="white"),  # 设置标题颜色为白色
        xaxis=dict(title_font=dict(color="white")),  # 设置x轴标签颜色为白色
        yaxis=dict(title_font=dict(color="white")),  # 设置y轴标签颜色为白色
        legend=dict(
            font=dict(
                family="Microsoft YaHei",
                color="white",
                size=14
            )
        )
    )
    fig_pie.update_traces(textfont=dict(color='white'))  # 确保标签字体颜色为白色
    st.plotly_chart(fig_pie, use_container_width=True)

# 中 - 地图
with center_col:
    fig_map = px.choropleth(
        state_fraud,
        locations='州',
        locationmode='USA-states',
        color='欺诈数量',
        scope="usa",
        title='各州欺诈交易分布',
        color_continuous_scale='Reds'
    )
    fig_map.update_layout(
        height=400,
        geo=dict(
            scope='usa',
            projection=dict(type='albers usa'),
            showcountries=False,
            showlakes=True,
            lakecolor='rgb(0,0,0)'
        ),
        margin=dict(t=50, b=0, l=0, r=0),
        template='plotly_dark',
        font=dict(family="Microsoft YaHei", color="white", size=14),
        paper_bgcolor='rgba(0,0,0,0)',  # 透明背景
        plot_bgcolor='rgba(0,0,0,0)',  # 透明背景
        title_font=dict(color="white"),  # 设置标题颜色为白色
        xaxis=dict(title_font=dict(color="white")),  # 设置x轴标签颜色为白色
        yaxis=dict(title_font=dict(color="white")),  # 设置y轴标签颜色为白色
        dragmode=False  # 禁止缩放和拖动，也可以设置为 'pan' 只允许平移

    )
    st.plotly_chart(fig_map, use_container_width=True)

# 右 - 柱状图
with right_col:
    fig_bar = px.bar(
        fraud_by_cat, x='欺诈数量', y='商户类别',
        orientation='h',
        title='欺诈交易最多的商户类别',
        color='欺诈数量',
        color_continuous_scale='Viridis'  # 更换颜色连续尺度
    )
    fig_bar.update_layout(
        height=320,
        template='plotly_dark',
        margin=dict(t=40, b=0, l=0, r=0),
        font=dict(family="Microsoft YaHei", color="white", size=14),
        paper_bgcolor='rgba(0,0,0,0)',  # 透明背景
        plot_bgcolor='rgba(0,0,0,0)',  # 透明背景
        title_font=dict(color="white"),  # 设置标题颜色为白色
        xaxis=dict(title_font=dict(color="white")),  # 设置x轴标签颜色为白色
        yaxis=dict(title_font=dict(color="white"))  # 设置y轴标签颜色为白色
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------
# 第二行：散点图 + 折线图
# -----------------------------
bot_left, bot_right = st.columns([1, 1])

with bot_left:
    if not df.empty:
        fig_scatter = px.scatter(
            df.sample(min(1500, len(df))), x='amt', y='city_pop', color='is_fraud',
            labels={'amt': '交易金额', 'city_pop': '城市人口', 'is_fraud': '是否欺诈'},
            title='交易金额与城市人口（按是否欺诈着色）',
            opacity=0.6,
            color_discrete_map={0: '#00FF7F', 1: '#FF4C4C'}  # 强烈颜色
        )
    else:
        fig_scatter = px.scatter(
            pd.DataFrame({'amt': [0], 'city_pop': [0], 'is_fraud': [0]}),
            x='amt', y='city_pop', color='is_fraud',
            labels={'amt': '交易金额', 'city_pop': '城市人口', 'is_fraud': '是否欺诈'},
            title='交易金额与城市人口（按是否欺诈着色）',
            opacity=0.6,
            color_discrete_map={0: '#00FF7F', 1: '#FF4C4C'}
        )
    fig_scatter.update_layout(
        height=330,
        template='plotly_dark',
        margin=dict(t=50, b=0, l=0, r=0),
        font=dict(family="Microsoft YaHei", color="white", size=14),
        paper_bgcolor='rgba(0,0,0,0)',  # 透明背景
        plot_bgcolor='rgba(0,0,0,0)',  # 透明背景
        title_font=dict(color="white"),  # 设置标题颜色为白色
        xaxis=dict(title_font=dict(color="white")),  # 设置x轴标签颜色为白色
        yaxis=dict(title_font=dict(color="white"))  # 设置y轴标签颜色为白色
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with bot_right:
    if not fraud_hour.empty:
        fig_line = px.line(
            fraud_hour, x='小时', y='欺诈数量',
            title='每小时欺诈交易趋势',
            markers=True,
            color_discrete_sequence=['#FF4C4C']  # 强烈的颜色
        )
    else:
        fig_line = px.line(
            pd.DataFrame({'小时': [0], '欺诈数量': [0]}),
            x='小时', y='欺诈数量',
            title='每小时欺诈交易趋势',
            markers=True,
            color_discrete_sequence=['#FF4C4C']
        )
    fig_line.update_layout(
        height=330,
        template='plotly_dark',
        margin=dict(t=50, b=0, l=0, r=0),
        font=dict(family="Microsoft YaHei", color="white", size=14),
        paper_bgcolor='rgba(0,0,0,0)',  # 透明背景
        plot_bgcolor='rgba(0,0,0,0)',  # 透明背景
        title_font=dict(color="white"),  # 设置标题颜色为白色
        xaxis=dict(title_font=dict(color="white")),  # 设置x轴标签颜色为白色
        yaxis=dict(title_font=dict(color="white"))  # 设置y轴标签颜色为白色
    )
    st.plotly_chart(fig_line, use_container_width=True)

# -----------------------------
# 新增图表：金额范围内的欺诈概率分布
# -----------------------------

# 设置金额范围（根据实际数据分布调整区间）
bins = [0, 10, 50, 100, 200, 500, 1000, 5000, 10000, 20000, np.inf]
labels = ['0-10', '10-50', '50-100', '100-200', '200-500', '500-1000', '1000-5000', '5000-10000', '10000-20000', '20000+']

# 确保labels数量比bins少1，正确的映射
if len(bins) - 1 != len(labels):
    raise ValueError(f"错误：'labels' 数量应比 'bins' 少一个。'bins' 有 {len(bins)}，而 'labels' 有 {len(labels)}")

# 根据交易金额划分区间
df['amt_range'] = pd.cut(df['amt'], bins=bins, labels=labels, right=False)  # right=False 确保上限不包含在内

# 计算每个金额范围内的欺诈交易数和总交易数
fraud_by_amt_range = df.groupby('amt_range')['is_fraud'].agg(['sum', 'count'])
fraud_by_amt_range['fraud_prob'] = fraud_by_amt_range['sum'] / fraud_by_amt_range['count'] if not fraud_by_amt_range['count'].empty else 0

# 创建图表
fig_freq_amt = px.bar(
    fraud_by_amt_range.reset_index(), x='amt_range', y='fraud_prob',
    title='不同金额范围内的欺诈概率分布',
    color='fraud_prob',
    color_continuous_scale='Reds'
)
fig_freq_amt.update_layout(
    height=320,
    template='plotly_dark',
    margin=dict(t=40, b=0, l=0, r=0),
    font=dict(family="Microsoft YaHei", color="white", size=14),
    paper_bgcolor='rgba(0,0,0,0)',  # 透明背景
    plot_bgcolor='rgba(0,0,0,0)',  # 透明背景
    title_font=dict(color="white"),  # 设置标题颜色为白色
    xaxis=dict(title_font=dict(color="white")),  # 设置x轴标签颜色为白色
    yaxis=dict(title_font=dict(color="white"))  # 设置y轴标签颜色为白色
)

# 在右列展示图表
with right_col:
    st.plotly_chart(fig_freq_amt, use_container_width=True)

# -----------------------------
# 新增图表：欺诈年龄分布
# -----------------------------
# 定义一个生成随机十六进制颜色的函数
# 生成随机颜色
def get_random_color():
    return f'rgba({random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)},{random.uniform(0.5, 1)})'

# 左 - 欺诈年龄分布
with left_col:
    if not fraud_age.empty:
        # 计算直方图数据
        hist_data = fraud_age.values
        bins = [i for i in range(int(min(fraud_age)), int(max(fraud_age)), (int(max(fraud_age)) - int(min(fraud_age))) // 20)]
        counts, _ = np.histogram(fraud_age, bins=bins)

        # 为每个柱子生成不同颜色
        colors = [get_random_color() for _ in range(len(counts))]

        # 创建图表
        fig_age = go.Figure()

        # 添加柱状图
        fig_age.add_trace(go.Bar(
            x=[f'{bins[i]} - {bins[i+1]}' for i in range(len(bins)-1)],
            y=counts,
            marker=dict(color=colors),
            showlegend=False
        ))
    else:
        fig_age = go.Figure()
        fig_age.add_trace(go.Bar(
            x=['0 - 0'],
            y=[0],
            marker=dict(color=get_random_color()),
            showlegend=False
        ))

    # 更新图表布局
    fig_age.update_layout(
        height=320,
        template='plotly_dark',
        title='欺诈交易年龄分布',
        margin=dict(t=40, b=0, l=0, r=0),
        font=dict(family="Microsoft YaHei", color="white", size=14),
        paper_bgcolor='rgba(0,0,0,0)',  # 透明背景
        plot_bgcolor='rgba(0,0,0,0)',  # 透明背景
        title_font=dict(color="white"),  # 设置标题颜色为白色
        xaxis=dict(title_font=dict(color="white")),  # 设置x轴标签颜色为白色
        yaxis=dict(title_font=dict(color="white"))  # 设置y轴标签颜色为白色
    )

    st.plotly_chart(fig_age, use_container_width=True)

# 中 - 欺诈月份分布
with center_col:
    if not fraud_month.empty:
        fig_month = px.bar(
            fraud_month, x='月份', y='欺诈数量',
            title='每月欺诈交易分布',
            color='欺诈数量',
            color_continuous_scale='Reds'
        )
    else:
        fig_month = px.bar(
            pd.DataFrame({'月份': [0], '欺诈数量': [0]}),
            x='月份', y='欺诈数量',
            title='每月欺诈交易分布',
            color='欺诈数量',
            color_continuous_scale='Reds'
        )
    fig_month.update_layout(
        height=320,
        template='plotly_dark',
        margin=dict(t=40, b=0, l=0, r=0),
        font=dict(family="Microsoft YaHei", color="white", size=14),
        paper_bgcolor='rgba(0,0,0,0)',  # 透明背景
        plot_bgcolor='rgba(0,0,0,0)',  # 透明背景
        title_font=dict(color="white"),  # 设置标题颜色为白色
        xaxis=dict(title_font=dict(color="white")),  # 设置x轴标签颜色为白色
        yaxis=dict(title_font=dict(color="white"))  # 设置y轴标签颜色为白色
    )
    st.plotly_chart(fig_month, use_container_width=True)

