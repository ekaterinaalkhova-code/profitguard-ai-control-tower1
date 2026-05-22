import io
import numpy as np
import pandas as pd


def read_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    return pd.read_excel(uploaded_file)


def clean_columns(df):
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("\n", " ", regex=False)
    )
    return df


def find_column(df, variants):
    columns = list(df.columns)

    for variant in variants:
        variant = variant.lower().strip()
        if variant in columns:
            return variant

    for column in columns:
        for variant in variants:
            if variant.lower().strip() in column:
                return column

    return None


def get_series(df, variants, default_value=None):
    col = find_column(df, variants)

    if col is None:
        return pd.Series([default_value] * len(df))

    return df[col]


def to_number(series):
    return (
        pd.to_numeric(series, errors="coerce")
        .fillna(0)
    )


def to_text(series):
    return (
        series
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
    )


def standardize_sales(df):
    df = clean_columns(df)

    result = pd.DataFrame()

    result["region"] = to_text(get_series(df, ["region", "регион", "филиал"], "UNKNOWN"))
    result["customer"] = to_text(get_series(df, ["customer", "клиент", "контрагент"], "UNKNOWN"))
    result["sku_code"] = to_text(get_series(df, ["sku_code", "артикул", "код", "код номенклатуры"], "UNKNOWN"))
    result["sku_name"] = to_text(get_series(df, ["sku_name", "товар", "номенклатура", "наименование"], "UNKNOWN"))
    result["brand"] = to_text(get_series(df, ["brand", "бренд"], "UNKNOWN"))
    result["supplier"] = to_text(get_series(df, ["supplier", "поставщик"], "UNKNOWN"))
    result["category"] = to_text(get_series(df, ["category", "категория", "дистр пакет", "пакет"], "UNKNOWN"))

    result["qty"] = to_number(get_series(df, ["qty", "количество", "кол-во", "шт"], 0))
    result["sales_amount"] = to_number(get_series(df, ["sales_amount", "продажа без ндс", "выручка", "оборот", "сумма"], 0))
    result["cogs"] = to_number(get_series(df, ["cogs", "себестоимость", "себест"], 0))
    result["gross_profit"] = to_number(get_series(df, ["gross_profit", "валовая прибыль", "вп", "прибыль"], 0))

    mask = (
        (result["gross_profit"] == 0)
        & (result["sales_amount"] != 0)
        & (result["cogs"] != 0)
    )

    result.loc[mask, "gross_profit"] = (
        result.loc[mask, "sales_amount"] - result.loc[mask, "cogs"]
    )

    return result


def standardize_stock(df):
    df = clean_columns(df)

    result = pd.DataFrame()

    result["region"] = to_text(get_series(df, ["region", "регион", "филиал"], "UNKNOWN"))
    result["warehouse"] = to_text(get_series(df, ["warehouse", "склад"], "UNKNOWN"))
    result["sku_code"] = to_text(get_series(df, ["sku_code", "артикул", "код", "код номенклатуры"], "UNKNOWN"))
    result["sku_name"] = to_text(get_series(df, ["sku_name", "товар", "номенклатура", "наименование"], "UNKNOWN"))
    result["brand"] = to_text(get_series(df, ["brand", "бренд"], "UNKNOWN"))
    result["supplier"] = to_text(get_series(df, ["supplier", "поставщик"], "UNKNOWN"))
    result["category"] = to_text(get_series(df, ["category", "категория", "дистр пакет", "пакет"], "UNKNOWN"))

    result["stock_qty"] = to_number(get_series(df, ["stock_qty", "остаток", "остаток шт", "количество"], 0))

    return result


def build_margin_engine(april_sales, may_sales):
    april = (
        april_sales
        .groupby(["sku_code", "sku_name", "brand", "supplier", "category"], as_index=False)
        .agg(
            april_qty=("qty", "sum"),
            april_sales=("sales_amount", "sum"),
            april_cogs=("cogs", "sum"),
            april_gp=("gross_profit", "sum")
        )
    )

    april["gp_per_unit_april"] = np.where(
        april["april_qty"] != 0,
        april["april_gp"] / april["april_qty"],
        0
    )

    april["cogs_per_unit_april"] = np.where(
        april["april_qty"] != 0,
        april["april_cogs"] / april["april_qty"],
        0
    )

    april["sales_price_per_unit_april"] = np.where(
        april["april_qty"] != 0,
        april["april_sales"] / april["april_qty"],
        0
    )

    brand_gp = (
        april
        .groupby("brand", as_index=False)
        .agg(brand_gp_per_unit=("gp_per_unit_april", "mean"))
    )

    supplier_gp = (
        april
        .groupby("supplier", as_index=False)
        .agg(supplier_gp_per_unit=("gp_per_unit_april", "mean"))
    )

    category_gp = (
        april
        .groupby("category", as_index=False)
        .agg(category_gp_per_unit=("gp_per_unit_april", "mean"))
    )

    result = (
        april
        .merge(brand_gp, on="brand", how="left")
        .merge(supplier_gp, on="supplier", how="left")
        .merge(category_gp, on="category", how="left")
    )

    result["final_gp_per_unit"] = result["gp_per_unit_april"]

    result["final_gp_per_unit"] = result["final_gp_per_unit"].replace(0, np.nan)
    result["final_gp_per_unit"] = result["final_gp_per_unit"].fillna(result["brand_gp_per_unit"])
    result["final_gp_per_unit"] = result["final_gp_per_unit"].fillna(result["supplier_gp_per_unit"])
    result["final_gp_per_unit"] = result["final_gp_per_unit"].fillna(result["category_gp_per_unit"])
    result["final_gp_per_unit"] = result["final_gp_per_unit"].fillna(0)

    may = (
        may_sales
        .groupby("sku_code", as_index=False)
        .agg(
            may_qty=("qty", "sum"),
            may_sales=("sales_amount", "sum")
        )
    )

    result = result.merge(may, on="sku_code", how="outer")

    result["may_qty"] = result["may_qty"].fillna(0)
    result["may_sales"] = result["may_sales"].fillna(0)
    result["may_calculated_gp"] = result["may_qty"] * result["final_gp_per_unit"]

    return result


def build_stock_control(stock, may_sales, margin_engine, target_days, max_days):
    sales_30 = (
        may_sales
        .groupby(["region", "sku_code"], as_index=False)
        .agg(
            sales_qty_30=("qty", "sum"),
            sales_amount_30=("sales_amount", "sum")
        )
    )

    stock_sum = (
        stock
        .groupby(
            ["region", "warehouse", "sku_code", "sku_name", "brand", "supplier", "category"],
            as_index=False
        )
        .agg(stock_qty=("stock_qty", "sum"))
    )

    result = stock_sum.merge(
        sales_30,
        on=["region", "sku_code"],
        how="left"
    )

    result["sales_qty_30"] = result["sales_qty_30"].fillna(0)
    result["sales_amount_30"] = result["sales_amount_30"].fillna(0)

    result["ads"] = result["sales_qty_30"] / 30

    result["doh"] = np.where(
        result["ads"] > 0,
        result["stock_qty"] / result["ads"],
        9999
    )

    price = margin_engine[
        [
            "sku_code",
            "final_gp_per_unit",
            "cogs_per_unit_april",
            "sales_price_per_unit_april"
        ]
    ].drop_duplicates("sku_code")

    result = result.merge(price, on="sku_code", how="left")

    result["final_gp_per_unit"] = result["final_gp_per_unit"].fillna(0)
    result["cogs_per_unit_april"] = result["cogs_per_unit_april"].fillna(0)
    result["sales_price_per_unit_april"] = result["sales_price_per_unit_april"].fillna(0)

    result["stock_value"] = result["stock_qty"] * result["cogs_per_unit_april"]

    result["target_stock_qty"] = result["ads"] * target_days
    result["max_stock_qty"] = result["ads"] * max_days

    result["overstock_qty"] = result["stock_qty"] - result["max_stock_qty"]
    result["overstock_qty"] = result["overstock_qty"].clip(lower=0)

    result["frozen_money"] = result["overstock_qty"] * result["cogs_per_unit_april"]

    conditions = [
        (result["stock_qty"] <= 0) | (result["doh"] < 3),
        (result["doh"] >= 3) & (result["doh"] < 10),
        (result["doh"] > max_days),
        (result["sales_qty_30"] == 0) & (result["stock_qty"] > 0)
    ]

    choices = [
        "CRITICAL_OOS",
        "RISK_OOS",
        "OVERSTOCK",
        "DEAD_STOCK"
    ]

    result["status"] = np.select(
        conditions,
        choices,
        default="NORMAL"
    )

    result["action"] = np.select(
        [
            result["status"] == "CRITICAL_OOS",
            result["status"] == "RISK_OOS",
            result["status"] == "OVERSTOCK",
            result["status"] == "DEAD_STOCK"
        ],
        [
            "Срочный заказ или переброска",
            "Проверить заказ и дату поставки",
            "Возврат / перемещение / промо",
            "Вывод / распродажа / списание"
        ],
        default="Контроль"
    )

    return result.sort_values(
        ["status", "frozen_money"],
        ascending=[True, False]
    )


def build_order_engine(stock_control, lead_time_days, safety_days):
    result = stock_control.copy()

    result["demand_during_lead_time"] = (
        result["ads"] * (lead_time_days + safety_days)
    )

    result["recommended_order_qty"] = (
        result["target_stock_qty"]
        + result["demand_during_lead_time"]
        - result["stock_qty"]
    )

    result["recommended_order_qty"] = result["recommended_order_qty"].clip(lower=0)

    result.loc[
        result["status"].isin(["OVERSTOCK", "DEAD_STOCK"]),
        "recommended_order_qty"
    ] = 0

    result["recommended_order_amount"] = (
        result["recommended_order_qty"] * result["cogs_per_unit_april"]
    )

    result["order_priority"] = np.select(
        [
            result["status"] == "CRITICAL_OOS",
            result["status"] == "RISK_OOS",
            result["status"] == "NORMAL"
        ],
        [
            "P1 Critical",
            "P2 Risk",
            "P3 Normal"
        ],
        default="No order"
    )

    return result[
        [
            "region",
            "warehouse",
            "supplier",
            "brand",
            "category",
            "sku_code",
            "sku_name",
            "status",
            "stock_qty",
            "sales_qty_30",
            "ads",
            "doh",
            "target_stock_qty",
            "demand_during_lead_time",
            "recommended_order_qty",
            "recommended_order_amount",
            "order_priority"
        ]
    ].sort_values(
        ["order_priority", "recommended_order_amount"],
        ascending=[True, False]
    )


def build_cash_flow(order_engine, purchase_budget):
    result = (
        order_engine
        .groupby(["region", "supplier", "brand"], as_index=False)
        .agg(
            recommended_order_amount=("recommended_order_amount", "sum"),
            recommended_order_qty=("recommended_order_qty", "sum")
        )
    )

    total_order = result["recommended_order_amount"].sum()

    if purchase_budget > 0:
        result["budget_share"] = result["recommended_order_amount"] / total_order
        result["allocated_budget"] = purchase_budget * result["budget_share"]
    else:
        result["budget_share"] = 0
        result["allocated_budget"] = 0

    result["budget_gap"] = result["recommended_order_amount"] - result["allocated_budget"]
    result["budget_gap"] = result["budget_gap"].clip(lower=0)

    result["cash_gap"] = result["budget_gap"]

    result["budget_status"] = np.select(
        [
            result["recommended_order_amount"] == 0,
            result["cash_gap"] == 0,
            result["cash_gap"] > 0
        ],
        [
            "NO_ORDER",
            "OK",
            "OVER_BUDGET"
        ],
        default="CHECK"
    )

    return result.sort_values("cash_gap", ascending=False)


def export_excel(sheets):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

            workbook = writer.book
            worksheet = writer.sheets[sheet_name[:31]]

            header_format = workbook.add_format({
                "bold": True,
                "bg_color": "#D9EAD3",
                "border": 1
            })

            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                worksheet.set_column(col_num, col_num, 18)

    return output.getvalue()
