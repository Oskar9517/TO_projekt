import requests
import tkinter as tk
from tkinter import ttk, messagebox
import ttkbootstrap as tb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
from scipy.stats import shapiro, pearsonr, skew, kurtosis
import numpy as np
import re
import math
import os

selected_correlation_vars = []

API_KEY = "aababab3-da9a-477f-108d-08dd8451a904"
BASE_URL = "https://bdl.stat.gov.pl/api/v1"
HEADERS = {
    "Accept": "application/json",
    "X-ClientId": API_KEY
}

def make_request(url, params=None):
    results = []
    page = 0
    while True:
        if params is None:
            params = {}
        params.update({"page": page, "page-size": 100})
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code != 200:
            messagebox.showerror("Błąd API", f"Błąd {response.status_code}: {response.text}")
            return []
        data = response.json()
        batch = data.get("results", [])
        results.extend(batch)
        if not data.get("links", {}).get("next"):
            break
        page += 1
    return results

def get_subjects(parent_id=None):
    if parent_id:
        url = f"{BASE_URL}/subjects?parent-id={parent_id}&lang=pl&format=json"
    else:
        url = f"{BASE_URL}/subjects?lang=pl&format=json"
    return make_request(url)

def get_variables(subject_id):
    return make_request(f"{BASE_URL}/variables?subject-id={subject_id}&lang=pl&format=json")

def get_available_years(var_id):
    data = make_request(f"{BASE_URL}/data/by-variable/{var_id}?format=json")
    years = sorted(set([item['year'] for unit in data for item in unit['values']]))
    return years

def get_data_by_variable(var_id, years=None):
    params = {"format": "json", "unit-level": 2}
    if years:
        params["year"] = years
    url = f"{BASE_URL}/data/by-variable/{var_id}"
    return make_request(url, params)
def normalize_name(name):
    replacements = {
        'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l',
        'ń': 'n', 'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z',
        'Ą': 'A', 'Ć': 'C', 'Ę': 'E', 'Ł': 'L',
        'Ń': 'N', 'Ó': 'O', 'Ś': 'S', 'Ź': 'Z', 'Ż': 'Z'
    }
    for pl, ascii_equiv in replacements.items():
        name = name.replace(pl, ascii_equiv)

    name = re.sub(r'[^\w\s]', '', name)
    return name.replace(' ', '_').lower()
    
def save_to_csv(data, var_id, filename, name_map=None):
    column_name = f"Wartość [{var_id}]"

    if name_map and column_name in name_map:
        readable_name = name_map[column_name]
    else:
        readable_name = column_name

    rows = []
    for unit in data:
        woj = unit["name"]
        for val in unit["values"]:
            rows.append({
                "Województwo": woj,
                "Rok": int(val["year"]),
                readable_name: val["val"]
            })

    df_new = pd.DataFrame(rows)
    df_new["Rok"] = df_new["Rok"].astype(int)
    df_new["Województwo"] = df_new["Województwo"].astype(str)

    if os.path.exists(filename):
        df_existing = pd.read_csv(filename)
        df_existing["Rok"] = df_existing["Rok"].astype(int)
        df_existing["Województwo"] = df_existing["Województwo"].astype(str)

        if readable_name in df_existing.columns:
            existing_years = set(df_existing[df_existing[readable_name].notna()]["Rok"].unique())
            new_years = set(df_new["Rok"].unique())
            overlapping_years = existing_years & new_years

            if overlapping_years == new_years:
                messagebox.showwarning(
                    "Zmienna już istnieje",
                    f"Zmienna dla wybranych lat znajduje się w pliku csv."
                )
                return False

            df_new = df_new[~df_new["Rok"].isin(overlapping_years)]

        df_combined = pd.merge(df_existing, df_new, on=["Rok", "Województwo"], how="outer")
    else:
        df_combined = df_new

    df_combined.sort_values(by=["Rok", "Województwo"], inplace=True)
    df_combined.to_csv(filename, index=False, encoding='utf-8-sig')
    return True
 
def interpret_r2(r2):
    if r2 < 0.3:
        return "słaba dopasowanie"
    elif r2 < 0.7:
        return "umiarkowane dopasowanie"
    else:
        return "silne dopasowanie"
def interpret_correlation(r):
    abs_r = abs(r)
    if abs_r >= 0.9:
        strength = "bardzo silna"
    elif abs_r >= 0.7:
        strength = "silna"
    elif abs_r >= 0.4:
        strength = "umiarkowana"
    elif abs_r >= 0.1:
        strength = "słaba"
    else:
        strength = "brak korelacji"

    direction = "dodatnia" if r > 0 else "ujemna" if r < 0 else ""
    
    if strength == "brak korelacji":
        return "Brak istotnej korelacji liniowej między zmiennymi."
    else:
        return f"{strength} {direction} korelacja — gdy jedna zmienna wzrasta, druga {'zwykle rośnie' if r > 0 else 'zwykle maleje'}."

def analyze_data_from_csv(csv_filename):
    if not os.path.exists(csv_filename):
        messagebox.showerror("Błąd", f"Plik ze zmiennymi nie istnieje.")
        return
    
    df = pd.read_csv(csv_filename)

    if df.empty:
        messagebox.showinfo("Analiza", "Brak danych do analizy.")
        return

    value_columns = df.columns[2:]
    for col in value_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if df.empty:
        messagebox.showinfo("Analiza", "Brak prawidłowych danych do analizy po oczyszczeniu.")
        return

    basic_info = []
    normality_tests = []
    correlations = []
    trends = []

    basic_info.append("=== Podstawowe informacje ===")
    basic_info.append(f"Liczba rekordów: {len(df)}")
    basic_info.append(f"Liczba województw: {df['Województwo'].nunique()}")
    basic_info.append(f"Zakres lat: {int(df['Rok'].min())} - {int(df['Rok'].max())}\n")

    normality_tests.append("=== Statystyki opisowe i test normalności ===")
    for col in value_columns:
        x = df[col].dropna()
        normality_tests.append(f"\nZmienna: {col}")
        normality_tests.append(f"  Średnia: {x.mean():.2f}")
        normality_tests.append(f"  Mediana: {x.median():.2f}")
        normality_tests.append(f"  Odchylenie standardowe: {x.std():.2f}")
        cv = x.std() / x.mean() if x.mean() != 0 else np.nan
        normality_tests.append(f"  Współczynnik zmienności: {cv:.2f}")
        normality_tests.append(f"  Skośność: {skew(x):.2f}")
        normality_tests.append(f"  Kurtoza: {kurtosis(x):.2f}")
        normality_tests.append(f"  Min: {x.min():.2f}, Max: {x.max():.2f}")
        if x.nunique() > 3:
            stat, p = shapiro(x.sample(min(len(x), 5000), random_state=1))
            normal = "TAK" if p > 0.05 else "NIE"
            normality_tests.append(f"  Normalność rozkładu: {normal} (p = {p:.4f})")
        else:
            normality_tests.append("  Za mało unikalnych wartości.")
        if "Województwo" in df.columns:
            df_filtered = df[["Rok", "Województwo", col]].dropna()
            percent_changes = {}

            for woj in df_filtered["Województwo"].unique():
                df_woj = df_filtered[df_filtered["Województwo"] == woj].sort_values("Rok")
                values = df_woj[col].values

                if len(values) < 2:
                    continue

                pct_changes = np.diff(values) / values[:-1]
                avg_pct_change = np.mean(pct_changes)

                percent_changes[woj] = avg_pct_change

            if percent_changes:
                max_woj = max(percent_changes, key=percent_changes.get)
                min_woj = min(percent_changes, key=percent_changes.get)
                max_val = percent_changes[max_woj] * 100
                min_val = percent_changes[min_woj] * 100

                normality_tests.append(f"  Największy średni roczny wzrost procentowy: {max_woj} ({max_val:.2f}%)")
                normality_tests.append(f"  Najniższy średni roczny wzrost procentowy: {min_woj} ({min_val:.2f}%)")
                
    trends.append("=== Trendy i prognoza ===")
    for col in value_columns:
        x = df['Rok'].values
        y = df[col].values
        mask = ~np.isnan(y)
        if mask.sum() < 3:
            trends.append(f"\nZmienna: {col} – zbyt mało danych do regresji.")
            continue

        x_clean = x[mask]
        y_clean = y[mask]
        slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)
        r2 = r_value ** 2
        trend_type = "Wzrostowy" if slope > 0 else "Spadkowy" if slope < 0 else "Stały"
        prediction_year = int(df['Rok'].max()) + 1
        prediction = intercept + slope * prediction_year
        equation = f"y = {slope:.2f} * x + {intercept:.2f}"

        direction_description = {
            "Wzrostowy": "Wartość tej zmiennej generalnie rosła w analizowanych latach.",
            "Spadkowy": "Wartość tej zmiennej generalnie malała w analizowanych latach.",
            "Stały": "Wartość tej zmiennej pozostała na zbliżonym poziomie w czasie."
        }

        if p_value < 0.01:
            confidence = "Bardzo wysoka pewność co do trendu."
        elif p_value < 0.05:
            confidence = "Wysoka pewność co do trendu."
        elif p_value < 0.1:
            confidence = "Umiarkowana pewność co do trendu."
        else:
            confidence = "Trend może być przypadkowy."

        region_info = ""
        if 'Województwo' in df.columns:
            avg_by_region = df.groupby("Województwo")[col].mean().dropna()
            if not avg_by_region.empty:
                top_region = avg_by_region.idxmax()
                top_value = avg_by_region.max()
                bottom_region = avg_by_region.idxmin()
                bottom_value = avg_by_region.min()
                region_info = (
                    f"Najwyższa wartość: {top_region} ({top_value:.2f})\n"
                    f"Najniższa wartość: {bottom_region} ({bottom_value:.2f})"
                )

        trends.append(f"\nZmienna: {col}")
        trends.append(f"Równanie trendu: {equation}")
        trends.append(f"R² = {r2:.2f} — {interpret_r2(r2)} linii trendu do danych.")
        trends.append(f"Typ trendu: {trend_type} - {direction_description[trend_type]}")
        trends.append(f"Istotność nachylenia: p = {p_value:.4f} — {'Nachylenie istotne statystycznie.' if p_value < 0.05 else 'Brak istotności statystycznej nachylenia.'} {confidence}")
        trends.append(f"Prognoza na rok {prediction_year} (średnia wartość): {prediction:.2f}")
        if region_info:
            trends.append(region_info)

    correlations.append("=== Korelacje między zmiennymi ===")

    window = tk.Toplevel()
    window.title("Zbiorczy raport analizy")
    window.geometry("1000x700")

    notebook = ttk.Notebook(window)
    notebook.pack(expand=True, fill='both')

    def create_tab(title, content, chart_buttons=None):
        frame = ttk.Frame(notebook)
        text = tk.Text(frame, wrap=tk.WORD, font=("Courier New", 10))
        text.insert(tk.END, "\n".join(content))
        text.config(state=tk.DISABLED)
        text.pack(expand=True, fill='both')

        if chart_buttons:
            button_frame = ttk.Frame(frame)
            button_frame.pack(fill='x')
            for name, chart_type in chart_buttons:
                def chart_callback(ct=chart_type):
                    if ct == "korelacje_wybor":
                        open_variable_selector_for_correlation(notebook)
                    else:
                        create_chart_from_csv(ct)
                ttk.Button(button_frame, text=name, command=chart_callback).pack(side='left', padx=5, pady=5)

        notebook.add(frame, text=title)

    create_tab("Informacje podstawowe", basic_info, [
        ("Wykres słupkowy", "słupkowy"),
        ("Wykres liniowy", "liniowy"),
        ("Wykres kołowy", "kołowy"),
        ("Heatmapa", "heatmapa"),
        ("Boxplot", "boxplot"),
        ("Wykres bąbelkowy", "bubble")
    ])
    create_tab("Statystyki i normalność", normality_tests, [("Top 5 województw", "top5_woj")])
    create_tab("Trendy i zmiany w czasie", trends, [
        ("Wykresy trendu i prognozy", "trendy_prognoza")
    ])

    create_tab("Korelacje", correlations, [
    ("Wybierz zmienne do korelacji", "korelacje_wybor"),
    ("Macierz korelacji", "korelacje_heatmapa")
])

def plot_correlation_heatmap(df, value_columns):
    corr_matrix = df[value_columns].corr(method='pearson')

    size = max(10, len(value_columns) * 1.5)
    fig, ax = plt.subplots(figsize=(size, size))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".6f",
        cmap="coolwarm",
        xticklabels=value_columns,
        yticklabels=value_columns,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax
    )

    ax.set_title("Macierz korelacji (Pearson)", fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def open_multi_selection_window(options):
    selected = []

    def add():
        val = combo.get()
        if val and val not in selected:
            selected.append(val)
            listbox.insert(tk.END, val)

    def remove():
        idx = listbox.curselection()
        if idx:
            selected.pop(idx[0])
            listbox.delete(idx)

    def done():
        win.destroy()

    win = tk.Toplevel()
    win.title("Wybierz zmienne do korelacji")

    combo = ttk.Combobox(win, values=options, state='readonly')
    combo.pack(padx=10, pady=5)

    ttk.Button(win, text="Dodaj", command=add).pack()
    listbox = tk.Listbox(win)
    listbox.pack(padx=10, pady=5)
    ttk.Button(win, text="Usuń", command=remove).pack()
    ttk.Button(win, text="Oblicz korelacje", command=done).pack(pady=10)
    win.wait_window()
    return selected

def create_correlation_output(notebook, new_lines):
    for tab in notebook.tabs():
        if notebook.tab(tab, "text") == "Korelacje":
            korelacje_tab = notebook.nametowidget(tab)
            text_widget = korelacje_tab.winfo_children()[0]
            
            text_widget.config(state='normal')

            full_text = text_widget.get("1.0", tk.END)
            header = "=== Korelacje między zmiennymi ==="
            header_index = full_text.find(header)

            if header_index != -1:
                line_start = full_text[:header_index].count('\n') + 1
                line_end = int(float(text_widget.index('end')))

                text_widget.delete(f"{line_start + 1}.0", tk.END)

            text_widget.insert(tk.END, "\n" + "\n".join(new_lines))
            text_widget.config(state='disabled')
            break

def create_chart_if_valid(chart_type):
    csv_filename = "variables.csv"
    if chart_type == "korelacje_wybor":
        open_variable_selector_for_correlation()
    else:
        result = create_chart_from_csv(csv_filename, chart_type)
        if result is False:
            print("Nie utworzono wykresu.")

def open_selection_window(df):
    window = tk.Toplevel()
    window.title("Wybierz województwo")

    selected_value = tk.StringVar()
    confirmed = {"ok": False}

    wojewodztwa = sorted(df["Województwo"].unique())
    selected_value.set(wojewodztwa[0])

    label = tk.Label(window, text="Województwo:")
    label.pack(padx=10, pady=5)

    dropdown = ttk.Combobox(window, textvariable=selected_value, values=wojewodztwa, state="readonly")
    dropdown.pack(padx=10, pady=5)

    def confirm_selection():
        confirmed["ok"] = True
        window.destroy()

    button = tk.Button(window, text="Zatwierdź", command=confirm_selection)
    button.pack(pady=10)

    def on_closing():
        confirmed["ok"] = False
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", on_closing)

    window.grab_set()
    window.wait_window()

    return selected_value.get() if confirmed["ok"] else None

def open_variable_selector_for_correlation(notebook):
    global selected_correlation_vars

    csv_filename = "variables.csv"
    selector = tk.Toplevel()
    selector.title("Wybierz zmienne do korelacji")
    selector.geometry("350x200")

    df = pd.read_csv(csv_filename)
    variable_columns = [col for col in df.columns if col not in ["Rok", "Województwo"]]

    if not variable_columns:
        print("Brak kolumn zmiennych w pliku.")
        return

    main_frame = ttk.Frame(selector)
    main_frame.pack(expand=True, fill="both", padx=10, pady=10)
    checkbox_frame = ttk.Frame(main_frame)
    checkbox_frame.pack(side="top", fill="both", expand=True)
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(side="bottom", fill="x")

    var_vars = []
    for col in variable_columns:
        var = tk.BooleanVar()
        chk = ttk.Checkbutton(checkbox_frame, text=col, variable=var)
        chk.pack(anchor='w')
        var_vars.append((col, var))

    def submit_selection():
        selected = [col for col, var in var_vars if var.get()]
        if len(selected) < 2:
            messagebox.showwarning("Błąd", "Wybierz co najmniej dwie zmienne.")
            return

        global selected_correlation_vars
        selected_correlation_vars = selected

        correlation_lines = []
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                col1, col2 = selected[i], selected[j]
                valid = df[[col1, col2]].dropna()
                if len(valid) >= 5:
                    r, p = pearsonr(valid[col1], valid[col2])
                    desc = interpret_correlation(r)
                    correlation_lines.append(f"Korelacja między {col1} a {col2}: r = {r:.2f} — {desc}")
                else:
                    correlation_lines.append(f"Korelacja między {col1} a {col2}: za mało danych.")

        create_correlation_output(notebook, correlation_lines)
        selector.destroy()

    ttk.Button(button_frame, text="Oblicz korelacje", command=submit_selection).pack(pady=5)

def create_chart_from_csv(chart_type):
    csv_filename = "variables.csv"
    df = pd.read_csv(csv_filename)

    variable_columns = [col for col in df.columns if col not in ["Rok", "Województwo"]]
    if not variable_columns:
        print("Brak kolumn zmiennych w pliku.")
        return

    df["Rok"] = df["Rok"].astype(int)
    df["Województwo"] = df["Województwo"].astype(str)

    if chart_type == "trendy_prognoza":
        selected_woj = open_selection_window(df)
        if selected_woj is None:
            return False

        for var_col in variable_columns:
            df_var = df[(df["Województwo"] == selected_woj)][["Rok", var_col]].dropna()
            if df_var.empty or df_var["Rok"].nunique() < 2:
                print(f"Za mało danych dla zmiennej {var_col} w województwie {selected_woj}.")
                continue

            x = df_var["Rok"].values.reshape(-1, 1)
            y = df_var[var_col].values

            model = LinearRegression()
            model.fit(x, y)
            next_year = df_var["Rok"].max() + 1
            y_pred = model.predict([[next_year]])[0]

            plt.figure(figsize=(10, 5))

            plt.plot(df_var["Rok"], y, marker='o', color='blue', label="Dane historyczne")

            plt.plot(
                [df_var["Rok"].values[-1], next_year],
                [y[-1], y_pred],
                marker='o',
                color='orange',
                label=f"Prognoza na {next_year}"
            )

            plt.title(f"{selected_woj} – {var_col}\nPrognoza na {next_year}")
            plt.xlabel("Rok")
            plt.ylabel("Wartość")
            plt.xticks(np.arange(df_var["Rok"].min(), next_year + 1, 1))
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
        return True
    elif chart_type == "top5_woj":
        if not variable_columns:
            print("Brak kolumn zmiennych do analizy.")
            return False

        for col in variable_columns:
            avg_by_region = df.groupby("Województwo")[col].mean().dropna().sort_values(ascending=False)

            top5 = avg_by_region.head(5)
            plt.figure(figsize=(8, 5))
            plt.barh(top5.index[::-1], top5.values[::-1], color='steelblue')
            plt.xlabel("Średnia wartość")
            plt.title(f"TOP 5 województw – {col}")
            for i, v in enumerate(top5.values[::-1]):
                plt.text(v, i, f"{v:.2f}", va='center')
            plt.tight_layout()
            plt.show()

        return True

    elif chart_type == "korelacje_heatmapa":
        global selected_correlation_vars
        if not selected_correlation_vars or len(selected_correlation_vars) < 2:
            messagebox.showinfo("Brak zmiennych", "Najpierw wybierz zmienne do analizy korelacji.")
            return False

        df = pd.read_csv("variables.csv")
        for col in selected_correlation_vars:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df_clean = df[selected_correlation_vars].dropna()

        if len(df_clean) < 5:
            messagebox.showinfo("Za mało danych", "Zbyt mało danych do wygenerowania macierzy korelacji.")
            return False

        plot_correlation_heatmap(df_clean, selected_correlation_vars)
        return True

    unique_years = sorted(df["Rok"].unique())
    for year in unique_years:
        df_year = df[df["Rok"] == year]

        non_empty_vars = [col for col in variable_columns if df_year[col].notna().any() and df_year[col].sum() != 0]
        if not non_empty_vars:
            continue

        num_vars = len(non_empty_vars)
        cols = min(num_vars, 3)
        rows = math.ceil(num_vars / cols)

        fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        axs = axs.flatten() if num_vars > 1 else [axs]

        for i, var_col in enumerate(non_empty_vars):
            ax = axs[i]
            ax.set_title(f"{var_col} ({year})", fontsize=12)

            if chart_type == "słupkowy":
                sns.barplot(x="Województwo", y=var_col, data=df_year, ax=ax)
                ax.tick_params(axis='x', rotation=90, labelsize=8)
            elif chart_type == "liniowy":
                sns.lineplot(x="Województwo", y=var_col, data=df_year, ax=ax, marker='o')
                ax.tick_params(axis='x', rotation=90, labelsize=8)
            elif chart_type == "boxplot":
                sns.boxplot(x="Województwo", y=var_col, data=df_year, ax=ax)
                ax.tick_params(axis='x', rotation=90, labelsize=8)
            elif chart_type == "kołowy":
                data = df_year[[var_col, "Województwo"]].dropna()
                if not data.empty:
                    ax.pie(data[var_col], labels=data["Województwo"], autopct='%1.1f%%', startangle=140)
                    ax.axis('equal')
            elif chart_type == "bubble":
                sizes = df_year[var_col].fillna(0) * 10
                sns.scatterplot(x="Województwo", y=var_col, size=sizes, sizes=(20, 300), data=df_year, ax=ax, legend=False)
                ax.tick_params(axis='x', rotation=90, labelsize=8)
            elif chart_type == "heatmapa":
                pivot = df_year.pivot(index="Województwo", columns="Rok", values=var_col)
                sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
            
            else:
                print(f"Nieznany typ wykresu: {chart_type}")
                return

            if chart_type != "kołowy":
                ax.set_xlabel("Województwo")
                ax.set_ylabel("Wartość")

        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        fig.suptitle(f"Wartości wg województw dla roku {year} ({chart_type})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

class BDLApp:
    def __init__(self, root):
        self.selected_correlation_vars = []
        self.root = root
        self.root.title("BDL - Wyszukiwarka danych")
        self.root.geometry("900x600")
        self.style = tb.Style('superhero')

        self.subjects = []
        self.subtopics1 = []
        self.subtopics2 = []
        self.variables = []
        self.years = []

        frame = tb.Frame(root, padding=20)
        frame.pack(fill='both', expand=True)

        self.subject_box = self.create_combobox(frame, "Wybierz temat główny:", self.subject_changed)
        self.subtopic1_box = self.create_combobox(frame, "Wybierz podtemat:", self.subtopic1_changed)
        self.subtopic2_box = self.create_combobox(frame, "Wybierz pod-podtemat:", self.subtopic2_changed)
        self.variable_box = self.create_combobox(frame, "Wybierz zmienną:", self.variable_changed)

        year_frame = tb.Frame(frame)
        year_frame.pack(pady=10)
        tb.Label(year_frame, text="Rok od:").pack(side='left', padx=5)
        self.year_from = ttk.Combobox(year_frame, state="readonly", width=10)
        self.year_from.pack(side='left', padx=5)
        tb.Label(year_frame, text="Rok do:").pack(side='left', padx=5)
        self.year_to = ttk.Combobox(year_frame, state="readonly", width=10)
        self.year_to.pack(side='left', padx=5)

        btn_frame = tb.Frame(frame)
        btn_frame.pack(pady=20)

        tb.Button(btn_frame, text="Dodaj zmienną", bootstyle="primary", command=self.save_variable_data_to_csv).pack(side='left', padx=10)
        tb.Button(btn_frame, text="Analiza danych", bootstyle="primary", command=lambda:analyze_data_from_csv("variables.csv")).pack(side='left', padx=10)
        self.load_subjects()

    def create_combobox(self, parent, label_text, command):
        tb.Label(parent, text=label_text).pack(pady=5)
        box = ttk.Combobox(parent, state="readonly")
        box.pack(fill='x', pady=5)
        if command:
            box.bind("<<ComboboxSelected>>", lambda e: command())
        return box

    def load_subjects(self):
        self.subjects = get_subjects()
        self.subject_box['values'] = [s['name'] for s in self.subjects]
    def subject_changed(self):
        idx = self.subject_box.current()
        if idx == -1:
            return
        selected = self.subjects[idx]
        self.subtopics1 = get_subjects(selected['id'])
        self.subtopic1_box['values'] = [s['name'] for s in self.subtopics1]
        self.subtopic1_box.set('')
        self.subtopics2 = []
        self.subtopic2_box.set('')
        self.variable_box.set('')
        self.variable_box['values'] = []
        self.clear_years()

    def subtopic1_changed(self):
        idx = self.subtopic1_box.current()
        if idx == -1:
            return
        selected = self.subtopics1[idx]
        self.subtopics2 = get_subjects(selected['id'])
        self.subtopic2_box['values'] = [s['name'] for s in self.subtopics2]
        self.subtopic2_box.set('')
        self.variable_box.set('')
        self.variable_box['values'] = []
        self.clear_years()

    def subtopic2_changed(self):
        idx = self.subtopic2_box.current()
        if idx == -1:
            return
        selected = self.subtopics2[idx]
        self.variables = get_variables(selected['id'])
        self.variable_box['values'] = [v['n1'] for v in self.variables]
        self.variable_box.set('')
        self.clear_years()

    def variable_changed(self):
        idx = self.variable_box.current()
        if idx == -1:
            return
        var_id = self.variables[idx]['id']
        self.years = get_available_years(var_id)
        if not self.years:
            self.year_from['values'] = []
            self.year_to['values'] = []
        else:
            self.year_from['values'] = self.years
            self.year_to['values'] = self.years
            self.year_from.set('')
            self.year_to.set('')
    
    def save_variable_data_to_csv(self):
        var_idx = self.variable_box.current()
        if var_idx == -1:
            messagebox.showwarning("Brak wyboru", "Wybierz zmienną.")
            return

        variable = self.variables[var_idx]
        var_id = variable['id']
        n1 = variable.get('n1', '').strip().lower()
        n2 = variable.get('n2', '').strip() if 'n2' in variable else None

        blacklist_prefixes = [
            "1 kwartał", "2 kwartał", "3 kwartał", "4 kwartał",
            "pierwsze półrocze", "drugie półrocze",
            "styczeń", "luty", "marzec", "kwiecień", "maj", "czerwiec",
            "lipiec", "sierpień", "wrzesień", "październik", "listopad", "grudzień"
        ]

        if any(n1.startswith(prefix) for prefix in blacklist_prefixes):
            messagebox.showwarning("Pominięto", "Wybrana zmienna nie zawiera danych rocznych.")
            return

        if n1 == "rok" and n2:
            full_name = n2
        elif n1 and n2:
            full_name = f"{n1} - {n2}"
        else:
            full_name = n1

        year_from = self.year_from.get()
        year_to = self.year_to.get()
        if not (year_from.isdigit() and year_to.isdigit()):
            messagebox.showerror("Błąd", "Podaj poprawne lata.")
            return
        year_from = int(year_from)
        year_to = int(year_to)
        if year_from > year_to:
            messagebox.showerror("Błąd", "Rok początkowy nie może być większy niż końcowy.")
            return
        years = list(range(year_from, year_to + 1))

        data = get_data_by_variable(var_id, years)
        if not data:
            messagebox.showerror("Błąd", "Brak danych.")
            return

        key = f"Wartość [{var_id}]"
        name_map = {key: normalize_name(full_name)}

        filename = "variables.csv"
        success = save_to_csv(data, var_id, filename, name_map)

        if success:
            messagebox.showinfo("Sukces", f"Dane zostały zapisane do pliku: {filename}")

    def clear_years(self):
        self.year_from['values'] = []
        self.year_to['values'] = []
        self.year_from.set('')
        self.year_to.set('')

if __name__ == "__main__":
    root = tb.Window(themename="superhero")
    app = BDLApp(root)
    root.mainloop()
