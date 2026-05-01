import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置页面配置
st.set_page_config(
    page_title="房地产市场数据分析",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 数据加载与清洗函数（复用原项目逻辑） ====================

@st.cache_data
def load_and_clean_data():
    """
    加载并清洗数据，复用原项目的数据处理逻辑
    """
    # 1. 导入数据
    df = pd.read_csv('data/house_sales.csv')

    # 2. 删除无用的数据列
    df.drop(columns='origin_url', inplace=True)

    # 3. 删除缺失值
    df.dropna(inplace=True)

    # 4. 删除重复值
    df.drop_duplicates(inplace=True)

    # 5. 数据类型转换（使用 .values.astype(str) 兼容 StringDtype）
    # 面积的数据类型转换
    df['area'] = pd.Series(df['area'].values.astype(str)).str.replace('㎡', '').astype(float)
    # 售价的数据类型转换
    df['price'] = pd.Series(df['price'].values.astype(str)).str.replace('万', '').astype(float)
    # 单价的数据类型转换
    df['unit'] = pd.Series(df['unit'].values.astype(str)).str.replace('元/㎡', '').astype(float)
    # 年份的数据类型转换
    df['year'] = pd.Series(df['year'].values.astype(str)).str.replace('年建', '').astype(int)
    # 朝向的数据类型转换
    df['toward'] = df['toward'].astype('category')

    # 6. 异常值处理
    # 房屋面积的异常处理
    df = df[(df["area"] < 600) & (df["area"] > 20)]
    # 房屋售价的异常处理 IQR
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    low_price = Q1 - 1.5 * IQR
    high_price = Q3 + 1.5 * IQR
    df = df[(df['price'] > low_price) & (df['price'] < high_price)]

    # 7. 数据特征构造
    # 地区 district
    df['district'] = df['address'].str.split('-').str[0]
    # 楼层的类型 floor_type
    df['floor_type'] = df['floor'].str.split('（').str[0].astype('category')
    # 是否是直辖市 is_zxs
    def fun1(str_val):
        if str_val in ['北京', '上海', '重庆', '天津']:
            return True
        else:
            return False
    df['is_zxs'] = df['city'].apply(fun1)
    # 卧室的数量 bedrooms
    df['bedrooms'] = df['rooms'].str.split('室').str[0].astype(int)
    # 客厅的数量 livingrooms
    df['livingrooms'] = df['rooms'].str.extract(r'(\d+)厅').astype(int)
    # 楼龄 building_age
    current_year = datetime.now().year
    df['building_age'] = current_year - df['year']
    # 价格的分段 price_label
    df['price_label'] = pd.cut(df['price'], bins=4, labels=['低价', '中价', '高价', '豪华'])

    return df


# ==================== Streamlit 页面主体 ====================

def main():
    # 侧边栏
    st.sidebar.title("🏠 房地产市场分析")
    st.sidebar.markdown("---")

    # 页面导航
    page = st.sidebar.radio(
        "选择页面",
        ["数据概览", "房价分析", "朝向分析", "面积分析"]
    )

    st.sidebar.markdown("---")
    st.sidebar.info("使用 Streamlit 构建的交互式数据分析面板")

    # 加载数据
    df = load_and_clean_data()

    # 顶部关键指标卡片（所有页面都显示）
    st.markdown("## 📊 关键指标")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="总房源数", value=f"{len(df):,}")
    with col2:
        st.metric(label="平均单价", value=f"{df['unit'].mean():.0f} 元/㎡")
    with col3:
        st.metric(label="平均总价", value=f"{df['price'].mean():.1f} 万")
    with col4:
        st.metric(label="平均面积", value=f"{df['area'].mean():.1f} ㎡")
    st.markdown("---")

    # 根据页面选择显示不同内容
    if page == "数据概览":
        show_data_overview(df)
    elif page == "房价分析":
        show_price_analysis(df)
    elif page == "朝向分析":
        show_toward_analysis(df)
    elif page == "面积分析":
        show_area_analysis(df)


# ==================== 数据概览页面 ====================

def show_data_overview(df):
    """数据概览页面：显示数据集的前10行数据和描述性统计"""
    st.markdown("## 📋 数据概览")

    # 显示数据集前10行
    st.subheader("数据集前10行")
    st.dataframe(df.head(10), use_container_width=True)

    # 显示描述性统计
    st.subheader("描述性统计")
    st.dataframe(df.describe(), use_container_width=True)

    # 显示数据基本信息
    st.subheader("数据基本信息")
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.metric("字段数量", len(df.columns))
    with info_col2:
        st.metric("数值型字段", len(df.select_dtypes(include=[np.number]).columns))
    with info_col3:
        st.metric("分类型字段", len(df.select_dtypes(include=['category', 'object']).columns))


# ==================== 房价分析页面 ====================

def show_price_analysis(df):
    """房价分析页面：显示房价分布直方图、箱线图，支持多维度筛选"""
    st.markdown("## 💰 房价分析")

    # 筛选条件
    st.sidebar.markdown("### 筛选条件")

    # 朝向筛选
    toward_options = ["全部"] + sorted(df['toward'].unique().tolist())
    selected_toward = st.sidebar.selectbox("选择朝向", toward_options)

    # 楼层筛选
    floor_options = ["全部"] + sorted(df['floor_type'].unique().tolist())
    selected_floor = st.sidebar.selectbox("选择楼层", floor_options)

    # 区域筛选（按城市）
    city_options = ["全部"] + sorted(df['city'].unique().tolist())
    selected_city = st.sidebar.selectbox("选择城市", city_options)

    # 应用筛选
    filtered_df = df.copy()
    if selected_toward != "全部":
        filtered_df = filtered_df[filtered_df['toward'] == selected_toward]
    if selected_floor != "全部":
        filtered_df = filtered_df[filtered_df['floor_type'] == selected_floor]
    if selected_city != "全部":
        filtered_df = filtered_df[filtered_df['city'] == selected_city]

    st.info(f"筛选后数据量：**{len(filtered_df)}** 条记录")

    # 图表类型选择
    chart_type = st.radio(
        "选择图表类型",
        ["房价分布直方图", "房价箱线图"],
        horizontal=True
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    if chart_type == "房价分布直方图":
        # 房价分布直方图
        sns.histplot(data=filtered_df, x='price', bins=30, kde=True, ax=ax, color='steelblue')
        ax.set_title('房价分布直方图', fontsize=16, fontweight='bold')
        ax.set_xlabel('房价（万）', fontsize=12)
        ax.set_ylabel('频数', fontsize=12)
    else:
        # 房价箱线图（按楼层类型）
        sns.boxplot(data=filtered_df, x='floor_type', y='price', ax=ax, palette='Set2')
        ax.set_title('不同楼层类型的房价箱线图', fontsize=16, fontweight='bold')
        ax.set_xlabel('楼层类型', fontsize=12)
        ax.set_ylabel('房价（万）', fontsize=12)

    plt.tight_layout()
    st.pyplot(fig)

    # 显示筛选后数据的统计信息
    st.subheader("筛选后数据统计")
    st.dataframe(filtered_df[['price', 'unit', 'area']].describe(), use_container_width=True)


# ==================== 朝向分析页面 ====================

def show_toward_analysis(df):
    """朝向分析页面：显示不同朝向的平均价格对比柱状图和统计表格"""
    st.markdown("## 🧭 朝向分析")

    # 计算不同朝向的平均价格
    toward_stats = df.groupby('toward').agg({
        'unit': ['mean', 'median', 'count'],
        'price': ['mean', 'median']
    }).round(2)
    toward_stats.columns = ['平均单价', '中位单价', '房源数量', '平均总价', '中位总价']
    toward_stats = toward_stats.reset_index()

    # 不同朝向的平均价格对比柱状图
    st.subheader("不同朝向的平均单价对比")
    fig, ax = plt.subplots(figsize=(12, 6))
    toward_mean = df.groupby('toward')['unit'].mean().sort_values(ascending=False)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(toward_mean)))
    bars = ax.bar(toward_mean.index, toward_mean.values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_title('不同朝向的平均单价对比', fontsize=16, fontweight='bold')
    ax.set_xlabel('朝向', fontsize=12)
    ax.set_ylabel('平均单价（元/㎡）', fontsize=12)
    ax.tick_params(axis='x', rotation=45)

    # 在柱子上添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)

    # 朝向价格统计表格
    st.subheader("朝向价格统计表格")
    st.dataframe(toward_stats, use_container_width=True)

    # 自动生成结论
    st.subheader("📌 分析结论")

    # 计算南北向和东西向的平均单价
    ns_towards = ['南北向', '南向北', '北向南', '北南向']
    ew_towards = ['东西向', '西向东', '东向西', '西东向']

    ns_prices = []
    ew_prices = []

    for t in df['toward'].unique():
        mean_price = df[df['toward'] == t]['unit'].mean()
        if any(keyword in str(t) for keyword in ['南北', '南', '北']):
            ns_prices.append(mean_price)
        elif any(keyword in str(t) for keyword in ['东西', '东', '西']):
            ew_prices.append(mean_price)

    if ns_prices and ew_prices:
        ns_avg = np.mean(ns_prices)
        ew_avg = np.mean(ew_prices)
        diff = ns_avg - ew_avg
        if diff > 0:
            st.success(f"**南北向**的房子平均单价比**东西向**贵 **{diff:.0f} 元/平方米**")
        else:
            st.success(f"**东西向**的房子平均单价比**南北向**贵 **{abs(diff):.0f} 元/平方米**")
    else:
        st.info("数据中没有足够的南北向或东西向样本进行对比分析。")


# ==================== 面积分析页面 ====================

def show_area_analysis(df):
    """面积分析页面：显示面积和价格的散点图"""
    st.markdown("## 📐 面积分析")

    st.subheader("面积与房价关系散点图")

    fig, ax = plt.subplots(figsize=(12, 7))

    # 使用不同颜色区分楼层类型
    scatter = ax.scatter(
        df['area'],
        df['price'],
        c=df['floor_type'].cat.codes,
        cmap='tab10',
        alpha=0.6,
        s=30,
        edgecolors='none'
    )

    ax.set_title('房屋面积与总价关系散点图', fontsize=16, fontweight='bold')
    ax.set_xlabel('面积（㎡）', fontsize=12)
    ax.set_ylabel('总价（万）', fontsize=12)

    # 添加趋势线
    z = np.polyfit(df['area'], df['price'], 1)
    p = np.poly1d(z)
    ax.plot(df['area'].sort_values(), p(df['area'].sort_values()),
            "r--", alpha=0.8, linewidth=2, label=f'趋势线')

    ax.legend(fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)

    # 显示相关系数
    corr = df['area'].corr(df['price'])
    st.info(f"面积与总价的皮尔逊相关系数：**{corr:.3f}**")

    # 面积分段统计
    st.subheader("面积分段统计")
    area_bins = [0, 50, 80, 120, 200, 1000]
    area_labels = ['<50㎡', '50-80㎡', '80-120㎡', '120-200㎡', '>200㎡']
    df['area_segment'] = pd.cut(df['area'], bins=area_bins, labels=area_labels)
    area_stats = df.groupby('area_segment', observed=True).agg({
        'price': 'mean',
        'unit': 'mean',
        'area': 'count'
    }).round(2)
    area_stats.columns = ['平均总价', '平均单价', '房源数量']
    st.dataframe(area_stats, use_container_width=True)


# ==================== 程序入口 ====================

if __name__ == "__main__":
    main()
