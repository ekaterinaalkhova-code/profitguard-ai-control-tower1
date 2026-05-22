import io
import pandas as pd
import streamlit as st

from core.calculations import (
    read_uploaded_file,
    standardize_sales,
    standardize_stock,
    build_margin_engine,
    build_stock_control,
    build_order_engine,
    build_cash_flow,
    export_excel,
)

st.set_page_config(
    page_title="ProfitGuard AI",
    page_icon="📦",
    layout="wide"
)

st.title("📦 ProfitGuard AI Control Tower")
st.caption("Остатки • OOS • Overstock • Автозаказ • Бюджет закупа • Cash Flow")

st.sidebar.header("Загрузка файлов")

april_file = st.sidebar.file_uploader(
    "1. Продажи апрель",
    type=["xlsx", "xls", "csv"]
)

may_file = st.sidebar.file_uploader(
    "2. Продажи май",
    type=["xlsx", "xls", "csv"]
)

stock_file = st.sidebar.file_uploader(
    "3. Остатки",
    type=["xlsx", "xls", "csv"]
)

st.sidebar.header("Параметры модели")

target_days = st.sidebar.number_input(
    "Целевой запас, дней",
    min_value=1,
    max_value=180,
    value=30
)

max_days = st.sidebar.number_input(
    "Overstock после, дней",
    min_value=1,
    max_value=365,
    value=60
)

lead_time_days = st.sidebar.number_input(
    "Средний срок поставки, дней",
    min_value=1,
    max_value=180,
    value=14
)

safety_days = st.sidebar.number_input(
    "Страховой запас, дней",
    min_value=0,
    max_value=90,
    value=7
)

purchase_budget = st.sidebar.number_input(
    "Бюджет закупа, ₸",
    min_value=0,
    value=50000000,
    step=1000000
)

run = st.sidebar.button("🚀 Рассчитать", type="primary")

if not run:
    st.info("Загрузите файлы слева и нажмите «Рассчитать».")
    st.markdown("""
    ### Минимум для работы:
    1. Продажи апрель  
    2. Продажи май  
    3. Остатки  

    ### Что считает модель:
    - валовую прибыль мая через апрельскую прибыль на штуку;
    - стоимость остатков;
    - дни запаса;
    - OOS;
    - overstock;
    - frozen money;
    - рекомендованный заказ;
    - превышение бюджета закупа;
    - cash-flow риск.
    """)
    st.stop()

if april_file is None or may_file is None or stock_file is None:
    st.error("Нужно загрузить три файла: продажи апрель, продажи май и остатки.")
    st.stop()

try:
    april_raw = read_uploaded_file(april_file)
    may_raw = read_uploaded_file(may_file)
    stock_raw = read_uploaded_file(stock_file)

    april_sales = standardize_sales(april_raw)
    may_sales = standardize_sales(may_raw)
    stock = standardize_stock(stock_raw)

    margin_engine = build_margin_engine(april_sales, may_sales)

    stock_control = build_stock_control(
        stock=stock,
        may_sales=may_sales,
        margin_engine=margin_engine,
        target_days=target_days,
        max_days=max_days
    )

    order_engine = build_order_engine(
        stock_control=stock_control,
        lead_time_days=lead_time_days,
        safety_days=safety_days
    )

    cash_flow = build_cash_flow(
        order_engine=order_engine,
        purchase_budget=purchase_budget
    )

    total_stock_value = stock_control["stock_value"].sum()
    frozen_money = stock_control["frozen_money"].sum()
    oos_sku = stock_control[stock_control["status"] == "CRITICAL_OOS"]["sku_code"].nunique()
    overstock_sku = stock_control[stock_control["status"] == "OVERSTOCK"]["sku_code"].nunique()
    recommended_order_amount = order_engine["recommended_order_amount"].sum()
    cash_gap = cash_flow["cash_gap"].sum()

    st.success("Расчет выполнен.")

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    k1.metric("Стоимость остатков", f"{total_stock_value:,.0f} ₸")
    k2.metric("Frozen money", f"{frozen_money:,.0f} ₸")
    k3.metric("OOS SKU", f"{oos_sku:,.0f}")
    k4.metric("Overstock SKU", f"{overstock_sku:,.0f}")
    k5.metric("Рекомендованный заказ", f"{recommended_order_amount:,.0f} ₸")
    k6.metric("Cash gap", f"{cash_gap:,.0f} ₸")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "CEO Dashboard",
        "Stock Control",
        "Order Engine",
        "Cash Flow",
        "Margin Engine"
    ])

    with tab1:
        st.subheader("Главные риски")

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Топ frozen money")
            st.dataframe(
                stock_control.sort_values("frozen_money", ascending=False).head(30),
                use_container_width=True
            )

        with col2:
            st.write("### OOS / Risk OOS")
            st.dataframe(
                stock_control[
                    stock_control["status"].isin(["CRITICAL_OOS", "RISK_OOS"])
                ].head(30),
                use_container_width=True
            )

    with tab2:
        st.subheader("Stock Control")
        st.dataframe(stock_control, use_container_width=True)

    with tab3:
        st.subheader("Order Engine")
        st.dataframe(order_engine, use_container_width=True)

    with tab4:
        st.subheader("Cash Flow")
        st.dataframe(cash_flow, use_container_width=True)

    with tab5:
        st.subheader("Margin Engine")
        st.dataframe(margin_engine, use_container_width=True)

    result_file = export_excel({
        "STOCK_CONTROL": stock_control,
        "ORDER_ENGINE": order_engine,
        "CASH_FLOW": cash_flow,
        "MARGIN_ENGINE": margin_engine
    })

    st.download_button(
        label="⬇ Скачать результат Excel",
        data=result_file,
        file_name="profitguard_ai_result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

except Exception as e:
    st.error("Ошибка расчета.")
    st.exception(e)
