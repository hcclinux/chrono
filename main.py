# -*- coding: utf-8 -*-
import sys
import re
import sqlite3
from pathlib import Path
from datetime import datetime, date

import pandas as pd
from PySide6.QtCore import QDate
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QLabel, QPushButton, QTextEdit, QGridLayout, QDateEdit,
    QComboBox, QLineEdit, QDialog, QDialogButtonBox, QListWidget,
    QListWidgetItem, QVBoxLayout
)

APP_NAME = "CSV 导入导出 · SQLite"
DB_PATH = Path("data.db")  # 也可做成可配置

# ---------- SQLite helpers ----------
def get_conn(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")  # 写入更稳
    return conn

def table_exists(conn, table):
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,))
    return cur.fetchone() is not None

def get_table_columns(conn, table):
    cur = conn.execute(f"PRAGMA table_info({table});")
    return [row[1] for row in cur.fetchall()]  # name col

def ensure_date_index(conn):
    conn.execute("CREATE INDEX IF NOT EXISTS idx_records_date ON records(date);")
    conn.commit()

# ---------- CSV import/export ----------
PREFERRED_ENCODINGS = ["utf-8", "utf-8-sig", "gb18030", "latin-1"]

def read_csv_any_encoding(csv_path, chunksize=None):
    last_err = None
    for enc in PREFERRED_ENCODINGS:
        try:
            if chunksize:
                return enc, pd.read_csv(csv_path, encoding=enc, chunksize=chunksize)
            else:
                return enc, pd.read_csv(csv_path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"无法读取 CSV，请确认编码/格式。最后错误：{last_err}")

def guess_date_column(df):
    # 优先中文“日期”
    candidates = [c for c in df.columns if str(c).strip().lower() in ("date", "日期", "交易日期", "day")]
    if "日期" in df.columns:
        return "日期"
    if candidates:
        return candidates[0]
    # 启发式：找看起来像 YYYYMMDD / YYYY-MM-DD 的列
    for col in df.columns:
        series = df[col].dropna().astype(str).head(20)
        if series.empty:
            continue
        score = 0
        for v in series:
            s = v.strip()
            if s.isdigit() and len(s) == 8:
                score += 1
                continue
            try:
                # 宽松解析
                _ = pd.to_datetime(s, errors="raise")
                score += 1
            except Exception:
                pass
        if score >= max(5, len(series) // 2):  # 半数以上像日期
            return col
    return None

def normalize_date_series(s):
    """
    将各种可能的日期形式转换为 'YYYY-MM-DD'(字符串)；无法解析则为 None
    """
    s = s.astype(str).str.strip()
    # 纯数字 YYYYMMDD
    is8digit = s.str.match(r"^\d{8}$", na=False)
    out = pd.Series([None] * len(s), index=s.index, dtype="object")

    if is8digit.any():
        tmp = pd.to_datetime(s[is8digit], format="%Y%m%d", errors="coerce")
        out.loc[is8digit] = tmp.dt.strftime("%Y-%m-%d")

    # 其它格式交给 to_datetime 自行识别
    rest = ~is8digit
    if rest.any():
        tmp = pd.to_datetime(s[rest], errors="coerce", dayfirst=False)
        out.loc[rest] = tmp.dt.strftime("%Y-%m-%d")

    # 标准化 None
    out = out.where(out.notna(), None)
    return out

def prepare_dataframe_for_import(df, date_col_guess="日期"):
    df = df.copy()

    date_col = date_col_guess if date_col_guess in df.columns else (guess_date_column(df) or "")
    if not date_col:
        raise RuntimeError("未能识别日期列，请确保有“日期”或“date”等列。")

    df["date"] = normalize_date_series(df[date_col])
    if df["date"].isna().all():
        raise RuntimeError(f"无法从列“{date_col}”解析出任何有效日期。")

    return df


def ensure_table_schema_and_align(conn, df, table="records", log=None):
    if log is None:
        log = print

    if not table_exists(conn, table):
        log("首次导入：创建表结构…")
        df.head(0).to_sql(table, conn, if_exists="replace", index=False)
        return

    existing_cols = set(get_table_columns(conn, table))
    for col in df.columns:
        if col not in existing_cols:
            conn.execute(f'ALTER TABLE {table} ADD COLUMN "{col}" TEXT;')
            existing_cols.add(col)

    for col in existing_cols:
        if col not in df.columns:
            df[col] = pd.NA


def append_prepared_dataframe(conn, df, table="records", log=None):
    if log is None:
        log = print

    ensure_table_schema_and_align(conn, df, table=table, log=log)
    log("写入数据库…")
    df.to_sql(table, conn, if_exists="append", index=False)
    ensure_date_index(conn)


def find_duplicate_dates(conn, dates, table="records"):
    if not table_exists(conn, table):
        return set()

    cleaned = [d for d in dates if d]
    if not cleaned:
        return set()

    unique_dates = list(dict.fromkeys(cleaned))
    duplicates = set()
    chunk_size = 500
    for i in range(0, len(unique_dates), chunk_size):
        chunk = unique_dates[i : i + chunk_size]
        placeholders = ",".join(["?"] * len(chunk))
        query = f"SELECT date FROM {table} WHERE date IN ({placeholders});"
        cur = conn.execute(query, chunk)
        duplicates.update(row[0] for row in cur.fetchall())
    return duplicates


def fetch_distinct_column_values(conn, column: str, limit: int = 1000) -> list[str]:
    if not table_exists(conn, "records"):
        return []
    existing_cols = set(get_table_columns(conn, "records"))
    if column not in existing_cols:
        return []
    query = f"""
    SELECT DISTINCT "{column}" FROM records
    WHERE "{column}" IS NOT NULL
    ORDER BY "{column}"
    LIMIT ?;
    """
    cur = conn.execute(query, (limit,))
    values = []
    for row in cur.fetchall():
        val = row[0]
        if val is None:
            continue
        values.append(str(val))
    return values


def append_dataframe(conn, df, table="records", date_col_guess="日期", log=None):
    prepared = prepare_dataframe_for_import(df, date_col_guess=date_col_guess)
    append_prepared_dataframe(conn, prepared, table=table, log=log)

def export_by_date_range(
    conn,
    start_date: str,
    end_date: str,
    out_csv: Path,
    search_column: str | None = None,
    search_keywords: list[str] | None = None,
):
    # 输入为 'YYYY-MM-DD'
    conditions = ["date >= ?", "date <= ?"]
    params: list[str] = [start_date, end_date]

    if search_column and search_keywords:
        existing_cols = set(get_table_columns(conn, "records")) if table_exists(conn, "records") else set()
        if search_column not in existing_cols:
            raise RuntimeError(f"数据表中未找到列“{search_column}”，无法筛选。")
        cleaned_keywords = [kw for kw in search_keywords if kw]
        if cleaned_keywords:
            like_clauses = []
            for kw in cleaned_keywords:
                like_clauses.append(f'"{search_column}" LIKE ?')
                params.append(f"%{kw}%")
            conditions.append("(" + " OR ".join(like_clauses) + ")")

    where_clause = " AND ".join(conditions)
    q = f"""
    SELECT * FROM records
    WHERE {where_clause}
    ORDER BY date ASC;
    """
    df = pd.read_sql_query(q, conn, params=params)
    if df.empty:
        return 0
    # 导出为带 BOM 的 UTF-8，Excel 友好
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return len(df)

# ---------- UI ----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(680, 420)

        self.conn = get_conn(DB_PATH)
        self.selected_keywords: list[str] = []

        # Widgets
        self.lbl_db = QLabel(f"数据库：{DB_PATH.resolve()}")
        self.btn_import = QPushButton("导入 CSV…")
        self.btn_export = QPushButton("按日期导出 CSV…")

        self.lbl_filter = QLabel("模糊搜索")
        self.combo_filter = QComboBox()
        self.combo_filter.addItem("代码", "代码")
        self.combo_filter.addItem("股票名字", "股票名字")
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("输入关键字，支持多个值（逗号、空格或换行分隔）")
        self.btn_pick_keywords = QPushButton("选择股票…")

        self.lbl_start = QLabel("起始日期")
        self.lbl_end = QLabel("结束日期")
        self.edit_start = QDateEdit(calendarPopup=True)
        self.edit_end = QDateEdit(calendarPopup=True)
        today = QDate.currentDate()
        self.edit_start.setDate(today.addMonths(-1))
        self.edit_end.setDate(today)

        self.log = QTextEdit()
        self.log.setReadOnly(True)

        # Layout
        root = QWidget()
        grid = QGridLayout(root)
        grid.addWidget(self.lbl_db, 0, 0, 1, 4)
        grid.addWidget(self.btn_import, 1, 0)
        grid.addWidget(self.lbl_start, 2, 0)
        grid.addWidget(self.edit_start, 2, 1)
        grid.addWidget(self.lbl_end, 3, 0)
        grid.addWidget(self.edit_end, 3, 1)
        grid.addWidget(self.lbl_filter, 4, 0)
        grid.addWidget(self.combo_filter, 4, 1)
        grid.addWidget(self.filter_edit, 4, 2)
        grid.addWidget(self.btn_pick_keywords, 4, 3)
        grid.addWidget(self.btn_export, 5, 0, 1, 4)
        grid.addWidget(self.log, 6, 0, 1, 4)
        grid.setRowStretch(6, 1)
        self.setCentralWidget(root)

        # Events
        self.btn_import.clicked.connect(self.on_import)
        self.btn_export.clicked.connect(self.on_export)
        self.btn_pick_keywords.clicked.connect(self.on_pick_keywords)
        self.combo_filter.currentIndexChanged.connect(self.on_filter_column_changed)

        # 初始化日期范围（如果库里已有数据）
        self.sync_date_limits()

    def logline(self, msg: str):
        self.log.append(msg)
        self.log.moveCursor(QTextCursor.End)

    def sync_date_limits(self):
        try:
            cur = self.conn.execute("SELECT MIN(date), MAX(date) FROM records;")
            row = cur.fetchone()
            if row and row[0] and row[1]:
                dmin = datetime.strptime(row[0], "%Y-%m-%d").date()
                dmax = datetime.strptime(row[1], "%Y-%m-%d").date()
                self.edit_start.setDate(QDate(dmin.year, dmin.month, dmin.day))
                self.edit_end.setDate(QDate(dmax.year, dmax.month, dmax.day))
                self.logline(f"库中日期范围：{row[0]} ～ {row[1]}")
        except Exception:
            pass

    def confirm_duplicate_dates(self, duplicates):
        preview = sorted(duplicates)
        if not preview:
            return True

        preview_text = "、".join(preview[:10])
        if len(preview) > 10:
            preview_text += f" …… 等 {len(preview)} 个日期"

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("重复日期确认")
        box.setText(f"以下日期已存在于数据库：\n{preview_text}\n是否继续导入？")
        confirm_btn = box.addButton("确认导入", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        return box.clickedButton() is confirm_btn

    def on_import(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择要导入的 CSV 文件", "", "CSV 文件 (*.csv)")
        if not path:
            return
        csv_path = Path(path)
        try:
            # 小于 100MB 直接整表读；更大可以用 chunksize 分块
            size_mb = csv_path.stat().st_size / (1024 * 1024)
            chunksize = 200_000 if size_mb > 100 else None
            enc, data = read_csv_any_encoding(csv_path, chunksize=chunksize)
            duplicates_confirmed = False

            def log_and_confirm_duplicates(dup_dates):
                if not dup_dates:
                    return True
                sorted_dates = sorted(dup_dates)
                preview = "、".join(sorted_dates[:10])
                if len(sorted_dates) > 10:
                    preview += f" …… 等 {len(sorted_dates)} 个日期"
                self.logline(f"检测到重复日期 {len(sorted_dates)} 个：{preview}")
                return self.confirm_duplicate_dates(sorted_dates)

            if isinstance(data, pd.io.parsers.readers.TextFileReader):
                total = 0
                for i, chunk in enumerate(data, 1):
                    prepared = prepare_dataframe_for_import(chunk, date_col_guess="日期")
                    duplicates = find_duplicate_dates(
                        self.conn,
                        prepared["date"].dropna().tolist(),
                    )
                    if duplicates and not duplicates_confirmed:
                        if not log_and_confirm_duplicates(duplicates):
                            self.logline("用户取消了导入。")
                            QMessageBox.information(self, "已取消", "检测到重复日期，导入已取消。")
                            return
                        duplicates_confirmed = True
                    append_prepared_dataframe(self.conn, prepared, table="records", log=self.logline)
                    total += len(prepared)
                    self.logline(f"已导入 {total} 行（分块 {i}）")
            else:
                prepared = prepare_dataframe_for_import(data, date_col_guess="日期")
                duplicates = find_duplicate_dates(
                    self.conn,
                    prepared["date"].dropna().tolist(),
                )
                if duplicates and not log_and_confirm_duplicates(duplicates):
                    self.logline("用户取消了导入。")
                    QMessageBox.information(self, "已取消", "检测到重复日期，导入已取消。")
                    return
                append_prepared_dataframe(self.conn, prepared, table="records", log=self.logline)
                self.logline(f"导入完成，共 {len(prepared)} 行（编码 {enc}）")

            self.sync_date_limits()
            QMessageBox.information(self, "完成", "CSV 导入完成。")
        except Exception as e:
            QMessageBox.critical(self, "导入失败", f"导入出错：\n{e}")

    def on_export(self):
        sd = self.edit_start.date().toPython()
        ed = self.edit_end.date().toPython()
        if sd > ed:
            QMessageBox.warning(self, "提示", "起始日期不能晚于结束日期。")
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "导出为 CSV", f"导出_{sd}_到_{ed}.csv", "CSV 文件 (*.csv)")
        if not out_path:
            return
        keywords = self.collect_keywords()
        search_column = self.combo_filter.currentData() if keywords else None
        try:
            n = export_by_date_range(
                self.conn,
                sd.isoformat(),
                ed.isoformat(),
                Path(out_path),
                search_column=search_column,
                search_keywords=keywords if keywords else None,
            )
            if n == 0:
                QMessageBox.information(self, "完成", "在该日期范围内没有数据。")
            else:
                QMessageBox.information(self, "完成", f"已导出 {n} 行到：\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"导出出错：\n{e}")

    def collect_keywords(self) -> list[str]:
        raw_text = self.filter_edit.text()
        if not raw_text.strip():
            self.selected_keywords = []
            return []
        manual = [
            part.strip()
            for part in re.split(r"[\s,，;；]+", raw_text)
            if part.strip()
        ]
        # 通过 dict.fromkeys 保持原有顺序同时去重
        merged = list(dict.fromkeys(manual))
        self.selected_keywords = merged
        return merged

    def on_filter_column_changed(self):
        self.selected_keywords.clear()
        self.filter_edit.clear()

    def on_pick_keywords(self):
        column = self.combo_filter.currentData()
        try:
            values = fetch_distinct_column_values(self.conn, column)
        except Exception as e:
            QMessageBox.critical(self, "读取失败", f"加载可选值时出错：\n{e}")
            return
        if not values:
            QMessageBox.information(self, "提示", "数据库中没有可选项，或尚未导入数据。")
            return
        dialog = KeywordPickerDialog(self, column, values, preselected=self.selected_keywords)
        if dialog.exec() == QDialog.Accepted:
            self.selected_keywords = dialog.selected_values()
            self.filter_edit.setText(", ".join(self.selected_keywords))


class KeywordPickerDialog(QDialog):
    def __init__(self, parent, column: str, values: list[str], preselected: list[str] | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"选择 {column}")
        self.resize(360, 420)

        layout = QVBoxLayout(self)
        instruction = QLabel("勾选需要导出的股票，可多选。")
        layout.addWidget(instruction)

        self.list_widget = QListWidget(self)
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)
        preselected_set = set(preselected or [])
        for value in values:
            item = QListWidgetItem(value)
            self.list_widget.addItem(item)
            if value in preselected_set:
                item.setSelected(True)
        layout.addWidget(self.list_widget)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def selected_values(self) -> list[str]:
        return [item.text() for item in self.list_widget.selectedItems()]


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
