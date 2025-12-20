# src/reports.py - Contains the Matplotlib PNG generation functions

import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta

# --- Trade-by-Trade Dashboard Generation ---
def generate_trade_by_trade_dashboard_png(df_filtered, R_value, display_mode, start_date, end_date):
    """Generates the Matplotlib PnL/Cumulative chart as a PNG in a buffer."""
    if df_filtered.empty:
        return None

    net_r = df_filtered['Display_PnL'].tolist()
    mfe_r = df_filtered['Display_MFE'].tolist()
    mae_r = df_filtered['Display_MAE'].tolist()
    cum_r = df_filtered['Display_PnL'].cumsum().tolist()

    unit_label = "R" if display_mode == "R" else "USD"

    # --- FIX: compare DATE only, not datetime ---
    start_d = pd.to_datetime(start_date).date()
    end_d = pd.to_datetime(end_date).date()

    if start_d == end_d:
        plot_date_range = start_d.strftime('%Y-%m-%d')
    else:
        plot_date_range = f"{start_d.strftime('%Y-%m-%d')} to {end_d.strftime('%Y-%m-%d')}"

    total_trades = len(net_r)

    # ---- STATS ----
    total_net_R = sum(net_r)
    total_mfe_R = sum(mfe_r)
    total_mae_R = sum(abs(val) for val in mae_r)
    
    mfe_mae_ratio = total_mfe_R / total_mae_R if total_mae_R != 0 else 0
    wins = [r for r in net_r if r > 0]
    losses = [r for r in net_r if r < 0]
    win_rate = len(wins) / total_trades * 100 if total_trades else 0
    avg_win = sum(wins)/len(wins) if wins else 0
    avg_loss = abs(sum(losses)/len(losses)) if losses else 0
    rr_ratio = avg_win / avg_loss if avg_loss != 0 else 0
    biggest_winner = max(net_r) if net_r else 0
    biggest_loser = min(net_r) if net_r else 0
    avg_r = sum(net_r)/total_trades if total_trades else 0

    # ---- DYNAMIC FIGURE HEIGHT ----
    base_height = 10
    extra_per_trade = 0.12
    fig_height = base_height + total_trades * extra_per_trade

    # ---- PLOT SETUP ----
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(14, fig_height),
        gridspec_kw={'height_ratios': [2, 1]}
    )

    x = list(range(1, total_trades + 1))

    # --- Top plot: MFE/MAE candlesticks ---
    for i, (r_val, mfe_val, mae_val) in enumerate(zip(net_r, mfe_r, mae_r), start=1):
        open_val = 0.0
        high_val = max(open_val, mfe_val, r_val)
        low_val = min(open_val, mae_val, r_val)
        ax1.vlines(i, low_val, high_val, color='black', linewidth=1, zorder=1)

    for i, r_val in enumerate(net_r, start=1):
        open_val = 0.0
        ax1.add_patch(plt.Rectangle(
            (i-0.2, min(open_val, r_val)),
            0.4,
            abs(r_val-open_val),
            facecolor='white' if r_val>=open_val else 'black',
            edgecolor='black',
            linewidth=1,
            zorder=2
        ))

    ax1.axhline(0, color='gray', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(i) for i in x], rotation=90)
    ax1.set_ylabel(f'P/L ({unit_label})')
    ax1.set_title(f'Trade-by-Trade VEN Dashboard ({unit_label}) – {plot_date_range}')
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # --- Bottom plot: Cumulative ---
    ax2.plot(x, cum_r, color='black', linewidth=2)
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(i) for i in x])
    ax2.set_ylabel(f'Cumulative {unit_label}')
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # --- Combined stats box ---
    combined_stats_text = (
        f"Total Trades: {total_trades}\n"
        f"Total Net {unit_label}: {total_net_R:.2f}\n"
        f"Win Rate: {win_rate:.1f}%\n"
        f"Avg W / Avg L: {rr_ratio:.2f}\n"
        f"Avg {unit_label}: {avg_r:.2f}\n"
        f"Avg W: {avg_win:.2f}  Avg L: {-avg_loss:.2f}\n"
        f"Max W: {biggest_winner:.2f}  Max L: {biggest_loser:.2f}\n\n"
        f"Total MFE ({unit_label}): {total_mfe_R:.2f}\n"
        f"Total MAE ({unit_label}): {-total_mae_R:.2f}\n"
        f"MFE / MAE Ratio: {mfe_mae_ratio:.2f}"
    )
    props = dict(boxstyle='round', facecolor='lightgray', alpha=0.9)
    ax1.text(1.02, 0.95, combined_stats_text, transform=ax1.transAxes,
             fontsize=13, verticalalignment='top', bbox=props, linespacing=1.4)

    # ---- TRADE LIST TEXT (RESTORED) ----
    max_per_column = 30
    trade_list_lines = [f"    ({unit_label})   MFE    MAE"]
    for i, (nr, mfe, mae) in enumerate(zip(net_r, mfe_r, mae_r), start=1):
        trade_list_lines.append(f"{i:<3}{nr:+.2f}  {mfe:.2f}  {mae:.2f}")

    if len(trade_list_lines) > max_per_column + 1:
        col1 = trade_list_lines[:max_per_column+1]
        col2 = trade_list_lines[max_per_column+1:]

        ax2.text(1.02, 1.0, "\n".join(col1), transform=ax2.transAxes,
                 fontsize=12, fontfamily="monospace", verticalalignment="top",
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9),
                 linespacing=1.4)

        ax2.text(1.35, 1.0, "\n".join(col2), transform=ax2.transAxes,
                 fontsize=12, fontfamily="monospace", verticalalignment="top",
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9),
                 linespacing=1.4)
    else:
        ax2.text(1.02, 0.95, "\n".join(trade_list_lines), transform=ax2.transAxes,
                 fontsize=12, fontfamily="monospace", verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9),
                 linespacing=1.4)

    # ---- FINAL LAYOUT ----
    plt.tight_layout(rect=[0, 0, 1.0, 1])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


# --- Weekly Dashboard Generation ---
def generate_weekly_dashboard_png(df_filtered, R_value, display_mode, start_date, end_date):
    """Generates the Matplotlib Weekly PnL/Cumulative chart as a PNG in a buffer."""
    if df_filtered.empty:
        return None

    net_r = df_filtered['Display_PnL'].tolist()
    mfe_r = df_filtered['Display_MFE'].tolist()
    mae_r = df_filtered['Display_MAE'].tolist()
    cum_r = df_filtered['Display_PnL'].cumsum().tolist()

    unit_label = "R" if display_mode == "R" else "USD"
    r_val_display = f" (R={R_value:.0f})" if display_mode == "R" else ""
    
    plot_date = f"{start_date.strftime('%Y-%m-%d')} \u2192 {end_date.strftime('%Y-%m-%d')}" 

    # ---- STATS ----
    total_net_R = float(sum(net_r))
    total_mfe_R = sum(mfe_r)
    total_mae_R = sum(abs(val) for val in mae_r)
    mfe_mae_ratio = total_mfe_R / total_mae_R if total_mae_R != 0 else 0
    total_trades = len(net_r)
    wins = [r for r in net_r if r > 0]
    losses = [r for r in net_r if r < 0]
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    avg_win = sum(wins)/len(wins) if wins else 0
    avg_loss = abs(sum(losses)/len(losses)) if losses else 0
    rr_ratio = avg_win / avg_loss if avg_loss != 0 else 0
    biggest_winner = max(net_r) if net_r else 0
    biggest_loser = min(net_r) if net_r else 0
    
    # ---- WEEKLY PER-DAY DATA ----
    df_filtered['Date'] = df_filtered['Entry DateTime'].dt.date
    daily_stats = df_filtered.groupby('Date').agg(
        Net_R=('Display_PnL', 'sum'),
        MFE_R=('Display_MFE', 'sum'),
        MAE_R=('Display_MAE', 'sum'), 
        Trades=('Display_PnL', 'count')
    ).reset_index()

    # ---- TRADE-LEVEL STREAKS ----
    wins_losses = df_filtered['Display_PnL'].apply(lambda x: 1 if x > 0 else -1)
    max_win_streak = 0
    max_loss_streak = 0
    current = 0
    for result in wins_losses:
        if result > 0:  # win
            current = current + 1 if current >= 0 else 1
            max_win_streak = max(max_win_streak, current)
        else:           # loss
            current = current - 1 if current <= 0 else -1
            max_loss_streak = min(max_loss_streak, current)
    trade_win_streak = max_win_streak
    trade_loss_streak = abs(max_loss_streak)


    # ---- FIGURE SETUP WITH GRIDSPEC ----
    fig = plt.figure(figsize=(16,16))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[2.5,1.5,1.75])

    # --- Top: trade-by-trade candlestick (ax1) ---
    ax1 = fig.add_subplot(gs[0, :])
    x = list(range(1, total_trades+1))
    
    for i, (r_val, mfe_val, mae_val) in enumerate(zip(net_r, mfe_r, mae_r), start=1):
        open_val = 0.0
        close_val = float(r_val)
        high_val = max(float(mfe_val), open_val, close_val)
        low_val = min(float(mae_val), open_val, close_val)

        ax1.vlines(i, low_val, high_val, color='black', linewidth=1, zorder=1)

        body_bottom = min(open_val, close_val)
        body_height = abs(close_val - open_val)

        ax1.add_patch(plt.Rectangle(
            (i - 0.25, body_bottom),
            0.5,
            body_height,
            facecolor='white' if close_val >= open_val else 'black',
            edgecolor='black',
            linewidth=1,
            zorder=2
        ))

    ax1.axhline(0, color='gray', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(i) for i in x], rotation=90)
    ax1.set_ylabel(f'P/L ({unit_label})')
    ax1.set_title(f'Weekly Trade VEN Dashboard ({unit_label}) – {plot_date}')
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # --- Stats box on top chart ---
    combined_stats_text = (
        f"Total Trades: {total_trades}\n"
        f"Total Net {unit_label}: {total_net_R:.2f}\n"
        f"Win Rate: {win_rate:.1f}%\n"
        f"Avg W / Avg L: {rr_ratio:.2f}\n"
        f"Avg {unit_label}: {total_net_R/total_trades:.2f}\n"
        f"Avg W: {avg_win:.2f}  Avg L: {-avg_loss:.2f}\n"
        f"Max W: {biggest_winner:.2f}  Max L: {biggest_loser:.2f}\n"
        f"W Streak: {trade_win_streak}\n"
        f"L Streak: {trade_loss_streak}\n\n"

        f"Total MFE ({unit_label}): {total_mfe_R:.2f}\n"
        f"Total MAE ({unit_label}): {-total_mae_R:.2f}\n"
        f"MFE / MAE Ratio: {mfe_mae_ratio:.2f}"
    )
    props = dict(boxstyle='round', facecolor='#d8d0e8', alpha=0.9)
    ax1.text(1.02, 0.95, combined_stats_text, transform=ax1.transAxes,
            fontsize=13, verticalalignment='top', bbox=props,
            linespacing=1.5)

    # --- Middle: cumulative R with alternating day shading (ax2) ---
    ax2 = fig.add_subplot(gs[1, :])
    x_plot = list(range(1, len(cum_r)+1))     
    ax2.plot(x_plot, cum_r, color='black', linewidth=2)

    # Determine day changes for shading - align with plot x indices
    dates = df_filtered['Date'].tolist()
    current_day = dates[0] if dates else None
    shade = True 
    start_idx = 0
    for i, d in enumerate(dates + [None]):
        if d != current_day or d is None:
            if shade:
                # Shade from trade index (start_idx + 1) up to (i)
                ax2.axvspan(start_idx + 0.5, i + 0.5, facecolor='lightgrey', alpha=0.3)
            start_idx = i
            current_day = d
            shade = not shade 

    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.set_xticks(x_plot)
    ax2.set_xticklabels([str(i) for i in x_plot], rotation=90)
    ax2.set_ylabel(f'Cumulative {unit_label}')
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # ---------------------------------------------------
    # METRICS: compute from ZERO-ANCHORED CUM ARRAY 
    # ---------------------------------------------------

    cum_for_metrics = [0.0] + list(cum_r)   
    cum_for_metrics = [float(v) for v in cum_for_metrics]

    peak = max(cum_for_metrics)
    valley = min(cum_for_metrics)

    # Max runup
    running_min = cum_for_metrics[0]
    max_runup = 0.0
    for val in cum_for_metrics:
        max_runup = max(max_runup, val - running_min)
        running_min = min(running_min, val)

    # Max drawdown
    running_max = cum_for_metrics[0]
    max_dd_val = 0.0
    for val in cum_for_metrics:
        max_dd_val = min(max_dd_val, val - running_max)
        running_max = max(running_max, val)
    max_dd_val = abs(max_dd_val)

    # ROMAD
    romad = total_net_R / abs(max_dd_val) if max_dd_val != 0.0 else float(total_net_R)
    
    cum_stats_text = (
        f"Peak: {peak:.2f}\n"
        f"Valley: {valley:.2f}\n"
        f"Max Runup: {max_runup:.2f}\n"
        f"Max Drawdown: {-max_dd_val:.2f}\n"
        f"T_ROMAD: {romad:.2f}"
    )
    props2 = dict(boxstyle='round', facecolor='#d8d0e8', alpha=0.9)
    ax2.text(1.02, 0.95, cum_stats_text, transform=ax2.transAxes,
            fontsize=13, verticalalignment='top', bbox=props2, linespacing=1.5)

    # --- Bottom left: weekly per-day candlestick (ax3) ---
    ax3 = fig.add_subplot(gs[2, 0])
    x_daily = list(range(1, len(daily_stats)+1))
    for i, (net_val, mfe_val, mae_val) in enumerate(zip(daily_stats['Net_R'], daily_stats['MFE_R'], daily_stats['MAE_R']), start=1):
        open_val = 0.0
        close_val = float(net_val)
        high_val = max(float(mfe_val), open_val, close_val)
        low_val = min(float(mae_val), open_val, close_val)

        ax3.vlines(i, low_val, high_val, color='black', linewidth=1, zorder=1)

        body_bottom = min(open_val, close_val)
        body_height = abs(close_val - open_val)

        ax3.add_patch(plt.Rectangle(
            (i - 0.25, body_bottom),
            0.5,
            body_height,
            facecolor='white' if close_val >= open_val else 'black',
            edgecolor='black',
            linewidth=1,
            zorder=2
        ))

    ax3.axhline(0, color='gray', linewidth=0.5)
    ax3.set_xticks(x_daily)
    ax3.set_xticklabels([d.strftime('%m-%d') for d in daily_stats['Date']], rotation=90)
    ax3.set_ylabel(f'P/L ({unit_label})')
    ax3.set_title('Weekly Per-Day Candlesticks (Total MFE/MAE)')
    ax3.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # --- Bottom right: weekly summary table (ax4) ---
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    table_data = daily_stats.copy()
    table_data['Date'] = pd.to_datetime(table_data['Date'])
    table_data['Date'] = table_data['Date'].dt.strftime('%a %d-%m-%Y') 

    table_data['Trades'] = table_data['Trades'].astype(int)
    table_data['Net_R'] = table_data['Net_R'].map('{:+.2f}'.format)
    table_data['MFE_R'] = table_data['MFE_R'].map('{:.2f}'.format)
    table_data['MAE_R'] = table_data['MAE_R'].map('{:.2f}'.format)

    columns = ['Date', 'Trades', 'Net_R', 'MFE_R', 'MAE_R']
    cell_text = table_data[columns].values.tolist()

    # --- Shading rows in table based on Net_R ---
    cell_colors = []
    table_data['Net_R_float'] = table_data['Net_R'].apply(lambda x: float(str(x).replace('+', ''))) 
    for _, row in table_data.iterrows():
        row_colors = []
        for col in columns:
            if col == 'Net_R' and row['Net_R_float'] < 0:
                row_colors.append('#e0e0e0')  
            else:
                row_colors.append('white')
        cell_colors.append(row_colors)

    table = ax4.table(
        cellText=cell_text,
        colLabels=[col if col != 'Net_R' else f'Net {unit_label}' for col in columns], 
        cellLoc='center',
        loc='center',
        bbox=[0.05, 0, 0.9, 1],
        cellColours=cell_colors,
        colColours=['#f9f4ff']*len(columns) 
    )

    # Adjust column widths
    n_rows = len(cell_text)
    for i in range(n_rows+1):
        table[i, 0].set_width(0.25)
        for j in range(1, 5):
            table[i, j].set_width(0.15)

    table.auto_set_font_size(True)
    table.scale(1, 1.2)

    # ---- FINAL LAYOUT ----
    plt.tight_layout(rect=[0, 0, 1.0, 1]) 

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf

# --- Monthly Dashboard Generation (NEW) ---
def generate_monthly_dashboard_png(df_filtered, R_value, display_mode, start_date, end_date):
    """Generates the Matplotlib Monthly PnL/Cumulative chart as a PNG in a buffer, 
    based on the monthly_dashboard_R.py style."""
    if df_filtered.empty:
        return None

    df_plot = df_filtered.copy()
    rolling_trades = 20  # rolling window for EV / Avg R

    net_r = df_plot['Display_PnL'].tolist()
    mfe_r = df_plot['Display_MFE'].tolist()
    mae_r = df_plot['Display_MAE'].tolist()
    cum_r = df_plot['Display_PnL'].cumsum().tolist()

    unit_label = "R" if display_mode == "R" else "USD"
    r_val_display = f" (R={R_value:.0f})" if display_mode == "R" else ""
    
    # ---- DATE RANGE ----
    # Ensure start/end date are datetime objects for formatting
    if not isinstance(start_date, datetime): start_date = datetime.combine(start_date, datetime.min.time())
    if not isinstance(end_date, datetime): end_date = datetime.combine(end_date, datetime.min.time())
    
    plot_date_title = f"{start_date.strftime('%Y-%m-%d')} \u2192 {end_date.strftime('%Y-%m-%d')}" 
    month_str_title = start_date.strftime('%B %Y') 

    # ---- STATS ----
    total_net_R = float(sum(net_r))
    total_mfe_R = sum(mfe_r)
    total_mae_R = sum(abs(val) for val in mae_r)
    mfe_mae_ratio = total_mfe_R / total_mae_R if total_mae_R != 0 else 0
    total_trades = len(net_r)
    wins = [r for r in net_r if r > 0]
    losses = [r for r in net_r if r < 0]

    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    avg_win = sum(wins)/len(wins) if wins else 0
    avg_loss = abs(sum(losses)/len(losses)) if losses else 0
    rr_ratio = avg_win / avg_loss if avg_loss != 0 else 0

    biggest_winner = max(net_r) if net_r else 0
    biggest_loser = min(net_r) if net_r else 0

    # ---- TRADE-LEVEL STREAKS ----
    wins_losses = df_plot['Display_PnL'].apply(lambda x: 1 if x > 0 else -1)

    max_win_streak = 0
    max_loss_streak = 0
    current = 0
    for result in wins_losses:
        if result > 0:  # win
            current = current + 1 if current >= 0 else 1
            max_win_streak = max(max_win_streak, current)
        else:           # loss
            current = current - 1 if current <= 0 else -1
            max_loss_streak = min(max_loss_streak, current)
    trade_win_streak = max_win_streak
    trade_loss_streak = abs(max_loss_streak)

    # ---- DAILY AGGREGATION FOR MONTH ----
    df_plot['Date'] = df_plot['Entry DateTime'].dt.date
    df_plot['Week'] = df_plot['Entry DateTime'].dt.isocalendar().week
    
    daily_stats = df_plot.groupby('Date').agg(
        Net_R=('Display_PnL', 'sum'),
        MFE_R=('Display_MFE', 'sum'),
        MAE_R=('Display_MAE', 'sum'), 
        Trades=('Display_PnL', 'count'),
        Week=('Week', 'first')
    ).reset_index()

    # Intraday cumulative
    df_plot['Daily_Cum_R'] = df_plot.groupby('Date')['Display_PnL'].cumsum()
    daily_intraday = df_plot.groupby('Date').agg(
        Intraday_High=('Daily_Cum_R', 'max'),
        Intraday_Low=('Daily_Cum_R', 'min')
    )
    daily_stats = daily_stats.merge(daily_intraday, on='Date', how='left')
    daily_stats['Cumulative_R'] = daily_stats['Net_R'].cumsum()

    # Calculate lagged weekly cumulative for daily plot
    weekly_sum = daily_stats.groupby('Week')['Net_R'].sum()
    weekly_cum_lagged = weekly_sum.shift(fill_value=0).cumsum()
    daily_stats['Weekly_Cum_R'] = daily_stats['Week'].map(weekly_cum_lagged)

    # ---- CALCULATE ROLLING EV / AVG R ----
    rolling_ev = df_plot['Display_PnL'].rolling(rolling_trades).mean()
    rolling_ev = rolling_ev.tolist()
    
    cum_net_r = df_plot['Display_PnL'].cumsum()
    trade_numbers = list(range(1, len(cum_net_r)+1))
    cum_avg_r = cum_net_r / trade_numbers

    # ---- FIGURE SETUP WITH GRIDSPEC ----
    fig = plt.figure(figsize=(18,22))  # Taller figure to fit EV
    # Layout: Top Trade-by-Trade, EV, Daily Cumulative, Daily Candles/Table
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[2.5,0.6,1.5,1.75])

    # --- Top: trade-by-trade candlestick (ax1) ---
    ax1 = fig.add_subplot(gs[0, :])
    x = list(range(1, total_trades+1))
    
    for i, (r_val, mfe_val, mae_val) in enumerate(zip(net_r, mfe_r, mae_r), start=1):
        open_val = 0.0
        close_val = float(r_val)
        high_val = max(float(mfe_val), open_val, close_val)
        low_val = min(float(mae_val), open_val, close_val)

        ax1.vlines(i, low_val, high_val, color='black', linewidth=1, zorder=1)

        body_bottom = min(open_val, close_val)
        body_height = abs(close_val - open_val)

        ax1.add_patch(plt.Rectangle(
            (i - 0.2, body_bottom),
            0.4,
            body_height,
            facecolor='white' if close_val >= open_val else 'black',
            edgecolor='black',
            linewidth=1,
            zorder=2
        ))

    ax1.axhline(0, color='gray', linewidth=0.5)
    ax1.set_xticks([])
    ax1.set_ylabel(f'P/L ({unit_label})')
    ax1.set_title(f'Monthly Trade VEN Dashboard ({unit_label}) – {month_str_title}')
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # --- Stats box on top chart ---
    combined_stats_text = (
        f"Total Trades: {total_trades}\n"
        f"Total Net {unit_label}: {total_net_R:.2f}\n"
        f"Win Rate: {win_rate:.1f}%\n"
        f"Avg W / Avg L: {rr_ratio:.2f}\n"
        f"Avg {unit_label}: {total_net_R/total_trades:.2f}\n"
        f"Avg W: {avg_win:.2f}  Avg L: {-avg_loss:.2f}\n"
        f"Max W: {biggest_winner:.2f}  Max L: {biggest_loser:.2f}\n"
        f"W Streak: {trade_win_streak}\n"
        f"L Streak: {trade_loss_streak}\n\n"
        f"Total MFE ({unit_label}): {total_mfe_R:.2f}\n"
        f"Total MAE ({unit_label}): {-total_mae_R:.2f}\n"
        f"MFE / MAE Ratio: {mfe_mae_ratio:.2f}"
    )

    props = dict(boxstyle='round', facecolor='#e8d0d0', alpha=0.9)
    ax1.text(1.02, 0.95, combined_stats_text, transform=ax1.transAxes,
            fontsize=13, verticalalignment='top', bbox=props, linespacing=1.5)

    # --- EV rolling subplot (ax_ev) ---
    ax_ev = fig.add_subplot(gs[1, :], sharex=ax1) 

    # Rolling EV 
    ax_ev.plot(trade_numbers, rolling_ev, color='blue', linewidth=2, label=f'{rolling_trades}-Trade Rolling Avg {unit_label}')

    # Cumulative avg R starting at trade 10
    min_trade_for_avg = 10
    ax_ev.plot(trade_numbers[min_trade_for_avg-1:], cum_avg_r[min_trade_for_avg-1:],
            color='orange', linewidth=2, linestyle='--', label=f'Cum Avg {unit_label}')

    ax_ev.axhline(0, color='gray', linewidth=0.5)
    ax_ev.set_ylabel(f'EV ({unit_label})')
    ax_ev.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # Set X-axis ticks
    max_trade = len(net_r)
    rough_step = max_trade // 10 if max_trade > 10 else 1
    nice_steps = [10, 20, 50, 100, 200, 500, 1000]
    step = min(n for n in nice_steps if n >= rough_step)
    tick_positions = list(range(0, max_trade+1, step))

    ax_ev.set_xticks(tick_positions)
    ax_ev.set_xticklabels(tick_positions)
    ax_ev.set_xlabel('Trade #')

    ax_ev.legend(loc='upper left')

    # --- Middle: Daily cumulative R (ax2) ---
    ax2 = fig.add_subplot(gs[2, :])
    x_daily = list(range(1, len(daily_stats)+1))

    ax2.plot(x_daily, daily_stats['Cumulative_R'], color='black', linewidth=2, label=f'Daily Cum {unit_label}')
    ax2.plot(x_daily, daily_stats['Weekly_Cum_R'], color='red',
            linestyle='--', linewidth=1.5, label=f'Weekly Cum {unit_label}', drawstyle='steps-post')

    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.set_xticks(x_daily)
    ax2.set_xticklabels([d.strftime('%d-%m') for d in daily_stats['Date']], rotation=90)
    ax2.set_ylabel(f'Daily Cumulative {unit_label}')
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.legend(loc='upper left')

    # Metrics for cum chart
    cum_for_metrics = [0.0] + daily_stats['Cumulative_R'].tolist()
    cum_for_metrics = [float(v) for v in cum_for_metrics]

    peak = max(cum_for_metrics)
    valley = min(cum_for_metrics)

    running_min = cum_for_metrics[0]
    max_runup = 0.0
    for val in cum_for_metrics:
        max_runup = max(max_runup, val - running_min)
        running_min = min(running_min, val)

    running_max = cum_for_metrics[0]
    max_dd = 0.0
    for val in cum_for_metrics:
        max_dd = min(max_dd, val - running_max)
        running_max = max(running_max, val)

    romad = total_net_R / abs(max_dd) if max_dd != 0.0 else float(total_net_R)

    cum_stats_text = (
        f"Peak: {peak:.2f}\n"
        f"Valley: {valley:.2f}\n"
        f"Max Runup: {max_runup:.2f}\n"
        f"Max Drawdown: {max_dd:.2f}\n"
        f"D_ROMAD: {romad:.2f}"
    )

    props2 = dict(boxstyle='round', facecolor='#e8d0d0', alpha=0.9)
    ax2.text(1.02, 0.95, cum_stats_text, transform=ax2.transAxes,
            fontsize=13, verticalalignment='top', bbox=props2, linespacing=1.5)

    # --- Bottom left: daily candles (ax3) ---
    ax3 = fig.add_subplot(gs[3, 0])

    for i, row in enumerate(daily_stats.itertuples(), start=1):
        open_val = 0.0
        close_val = row.Net_R
        high_val = max(open_val, row.Intraday_High, close_val)
        low_val = min(open_val, row.Intraday_Low, close_val)

        ax3.vlines(i, low_val, high_val, color='black', linewidth=1, zorder=1)

    for i, net_val in enumerate(daily_stats['Net_R'], start=1):
        open_val = 0.0
        ax3.add_patch(plt.Rectangle(
            (i-0.2, min(open_val, net_val)),
            0.4,
            abs(net_val-open_val),
            facecolor='white' if net_val>=open_val else 'black',
            edgecolor='black',
            linewidth=1,
            zorder=2
        ))

    ax3.axhline(0, color='gray', linewidth=0.5)
    ax3.set_xticks(x_daily)
    ax3.set_xticklabels([d.strftime('%d-%m') for d in daily_stats['Date']], rotation=90)
    ax3.set_ylabel(f'P/L ({unit_label})')
    ax3.set_title(f'{month_str_title} Daily Candlesticks (Closed P/L)')
    ax3.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # --- Bottom right: daily stats table (ax4) ---
    ax4 = fig.add_subplot(gs[3, 1])
    ax4.axis('off')

    table_data = daily_stats.copy()
    table_data['Date'] = pd.to_datetime(table_data['Date']).dt.strftime('%a %d-%m-%Y')
    table_data['Trades'] = table_data['Trades'].astype(int)
    
    # Create display columns for formatting but use original Net_R for logic
    table_data['Net_R_display'] = table_data['Net_R'].map('{:+.2f}'.format)
    table_data['MFE_R_display'] = table_data['MFE_R'].map('{:.2f}'.format)
    table_data['MAE_R_display'] = table_data['MAE_R'].map('{:.2f}'.format)

    columns = ['Date', 'Trades', 'Net_R_display', 'MFE_R_display', 'MAE_R_display']
    column_labels = ['Date', 'Trades', f'Net {unit_label}', f'MFE {unit_label}', f'MAE {unit_label}']
    cell_text = table_data[columns].values.tolist()

    header_bg_color = '#fff0f0'
    cell_colors = []
    # Use the original Net_R for coloring decision
    for _, row in daily_stats.iterrows():
        row_colors = []
        for col in columns:
            if col == 'Net_R_display' and row['Net_R'] < 0:
                row_colors.append('#e0e0e0')
            else:
                row_colors.append('white')
        cell_colors.append(row_colors)

    table = ax4.table(
        cellText=cell_text,
        colLabels=column_labels,
        cellLoc='center',
        loc='center',
        bbox=[0.05, 0, 0.9, 1],
        cellColours=cell_colors
    )

    n_rows = len(cell_text)
    for i in range(n_rows+1):
        table[i, 0].set_width(0.25)
        for j in range(1, 5):
            table[i, j].set_width(0.15)

    table.auto_set_font_size(True)
    table.scale(1, 1.2)
    for j, col in enumerate(column_labels):
        table[0, j].set_facecolor(header_bg_color)

    # Stats box for bottom table
    total_days = len(daily_stats)
    win_days = daily_stats[daily_stats['Net_R'] > 0]
    loss_days = daily_stats[daily_stats['Net_R'] < 0]

    win_rate_days = len(win_days) / total_days * 100 if total_days > 0 else 0
    avg_win_day = win_days['Net_R'].mean() if not win_days.empty else 0
    avg_loss_day = abs(loss_days['Net_R'].mean()) if not loss_days.empty else 0
    rr_day = avg_win_day / avg_loss_day if avg_loss_day != 0 else 0

    best_day = daily_stats['Net_R'].max() if total_days > 0 else 0.0
    worst_day = daily_stats['Net_R'].min() if total_days > 0 else 0.0
    avg_day = daily_stats['Net_R'].mean() if total_days > 0 else 0.0

    daily_stats_text = (
        f"Total Days: {total_days}\n"
        f"Total Net {unit_label}: {total_net_R:.2f}\n"
        f"Win% Days: {win_rate_days:.1f}%\n"
        f"Avg W / Avg L Day: {rr_day:.2f}\n"
        f"Avg Day {unit_label}: {avg_day:.2f}\n"
        f"Avg W Day: {avg_win_day:.2f}\n"
        f"Avg L Day: {-avg_loss_day:.2f}\n"
        f"Best Day: {best_day:.2f}\n"
        f"Worst Day: {worst_day:.2f}"
    )

    props_daily = dict(boxstyle='round', facecolor='#e8d0d0', alpha=0.9)
    ax4.text(1.02, 0.95, daily_stats_text, transform=ax4.transAxes,
             fontsize=13, verticalalignment='top', bbox=props_daily, linespacing=1.5)

    # ---- FINAL LAYOUT ----
    plt.tight_layout()

    # Manually adjust gap between ax1 (top chart) and ax_ev (rolling EV)
    pos1 = ax1.get_position()
    pos_ev = ax_ev.get_position()

    new_y1 = pos1.y0
    new_y0 = pos_ev.y0
    new_height = new_y1 - new_y0

    ax_ev.set_position([pos_ev.x0, new_y0, pos_ev.width, new_height])

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf

    # --- Quarterly Dashboard Generation (NEW) ---
def generate_quarterly_dashboard_png(df_filtered, R_value, display_mode, start_date, end_date):
    """Generates the Quarterly Matplotlib chart based on quarterly_dashboard_USD.py,
    dynamically adapting to USD or R units via display_mode."""
    if df_filtered.empty:
        return None

    df_plot = df_filtered.copy()
    rolling_trades = 20
    
    unit_label = "R" if display_mode == "R" else "USD"
    # Decide if we prepend a dollar sign (only for USD mode, typically)
    currency_symbol = "$" if display_mode == "USD" else ""
    
    # Map standardized columns to variables
    net_val = df_plot['Display_PnL'].tolist()
    mfe_val = df_plot['Display_MFE'].tolist()
    mae_val = df_plot['Display_MAE'].tolist()
    
    # ---- DATES ----
    if not isinstance(start_date, datetime): start_date = datetime.combine(start_date, datetime.min.time())
    if not isinstance(end_date, datetime): end_date = datetime.combine(end_date, datetime.min.time())
    
    # Calculate Quarter labels
    def get_quarter(dt): return (dt.month - 1) // 3 + 1
    q_start = get_quarter(start_date)
    q_end = get_quarter(end_date)
    y_start = start_date.year
    y_end = end_date.year

    if (q_start == q_end) and (y_start == y_end):
        plot_title = f'Quarterly Trade VEN Dashboard ({unit_label}) – Q{q_start} {y_start}'
    else:
        plot_title = f'Quarterly Trade VEN Dashboard ({unit_label}) – Q{q_start} {y_start} to Q{q_end} {y_end}'

    # ---- STATS ----
    total_net = sum(net_val)
    total_mfe = sum(mfe_val)
    total_mae = sum(abs(v) for v in mae_val)
    mfe_mae_ratio = (total_mfe / total_mae) if total_mae != 0 else 0

    total_trades = len(net_val)
    wins = [v for v in net_val if v > 0]
    losses = [v for v in net_val if v < 0]
    
    total_wins = len(wins)
    total_losses = len(losses)
    total_wins_val = sum(wins)
    total_losses_val = sum(losses)

    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    avg_win = sum(wins)/len(wins) if wins else 0
    avg_loss = abs(sum(losses)/len(losses)) if losses else 0
    rr_ratio = (avg_win / avg_loss) if avg_loss != 0 else 0

    biggest_winner = max(net_val) if net_val else 0
    biggest_loser = min(net_val) if net_val else 0

    # ---- STREAKS ----
    wins_losses = df_plot['Display_PnL'].apply(lambda x: 1 if x > 0 else -1)
    max_win_streak = 0
    max_loss_streak = 0
    current = 0
    for result in wins_losses:
        if result > 0:
            current = current + 1 if current >= 0 else 1
            max_win_streak = max(max_win_streak, current)
        else:
            current = current - 1 if current <= 0 else -1
            max_loss_streak = min(max_loss_streak, current)
    trade_win_streak = max_win_streak
    trade_loss_streak = abs(max_loss_streak)

    # ---- DAILY AGGREGATION ----
    df_plot['Date'] = df_plot['Entry DateTime'].dt.date
    df_plot['Week'] = df_plot['Entry DateTime'].dt.isocalendar().week
    df_plot['MonthNum'] = df_plot['Entry DateTime'].dt.month

    daily_stats = df_plot.groupby('Date').agg(
        Net=('Display_PnL', 'sum'),
        MFE=('Display_MFE', 'sum'),
        MAE=('Display_MAE', 'sum'),
        Trades=('Display_PnL', 'count'),
        Week=('Week', 'first'),
        MonthNum=('MonthNum', 'first')
    ).reset_index()

    # Intraday cumulative path for daily candles
    df_plot['Daily_Cum'] = df_plot.groupby('Date')['Display_PnL'].cumsum()
    daily_intraday = df_plot.groupby('Date').agg(
        Intraday_High=('Daily_Cum', 'max'),
        Intraday_Low=('Daily_Cum', 'min')
    )
    daily_stats = daily_stats.merge(daily_intraday, on='Date', how='left')
    daily_stats['Cumulative'] = daily_stats['Net'].cumsum()

    # Weekly lagged cumulative for daily chart
    weekly_sum = daily_stats.groupby('Week')['Net'].sum()
    weekly_cum_lagged = weekly_sum.shift(fill_value=0).cumsum()
    daily_stats['Weekly_Cum'] = daily_stats['Week'].map(weekly_cum_lagged)

    # ---- ROLLING EV / AVG ----
    rolling_ev = df_plot['Display_PnL'].rolling(rolling_trades).mean().tolist()
    cum_net = df_plot['Display_PnL'].cumsum()
    trade_numbers = list(range(1, len(cum_net)+1))
    cum_avg = (cum_net / trade_numbers).tolist()

    # ---- FIGURE SETUP ----
    fig = plt.figure(figsize=(22,24))
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[2.5,0.6,1.5,1.75])

    # --- Top: Scatter Bubble Chart (ax1) ---
    ax1 = fig.add_subplot(gs[0, :])
    x_trades = list(range(1, len(net_val)+1))
    
    # Circle areas proportional to absolute P/L
    sizes = np.array([abs(x) for x in net_val])
    max_size = 500
    sizes_scaled = sizes / sizes.max() * max_size if sizes.max() > 0 else sizes
    colors = ['green' if val >= 0 else 'red' for val in net_val]

    ax1.scatter(x_trades, net_val, s=sizes_scaled, c=colors, alpha=0.6, edgecolors='none', linewidth=0.5)
    ax1.axhline(0, color='gray', linewidth=0.5)

    # Monthly shading (alternating grey/white)
    unique_months = df_plot['MonthNum'].unique()
    for i, month in enumerate(unique_months):
        if i % 2 == 0:
            month_mask = df_plot['MonthNum'] == month
            if month_mask.any():
                # Indices for shading
                start_idx = month_mask.idxmax() + 1 - 0.5
                end_idx = month_mask[::-1].idxmax() + 1 + 0.5
                ax1.axvspan(start_idx, end_idx, color='lightgrey', alpha=0.2, zorder=0)

    ax1.set_xticks([])
    ax1.set_ylabel(f'P/L ({unit_label})')
    ax1.set_title(plot_title)
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # --- Stats Box Top ---
    # Helper to format currency/units
    def fmt(val): return f"{currency_symbol}{val:,.2f}"

    combined_stats_text = (
        f"Total Trades: {total_trades}\n"
        f"Total Net P/L: {fmt(total_net)}\n"
        f"Win Rate: {win_rate:.1f}%\n"
        f"Avg W / Avg L: {rr_ratio:.2f}\n"
        f"Avg trade: {fmt(sum(net_val)/total_trades) if total_trades>0 else 0}\n"
        f"Avg W: {fmt(avg_win)}\n"  
        f"Avg L: {fmt(-avg_loss)}\n"
        f"Max W: {fmt(biggest_winner)}\n"
        f"Max L: {fmt(biggest_loser)}\n\n"
        f"W Trades: {total_wins}\n"
        f"L Trades: {total_losses}\n"
        f"W Streak: {trade_win_streak}\n"
        f"L Streak: {trade_loss_streak}\n\n"
        f"Total W P/L: {fmt(total_wins_val)}\n"
        f"Total L P/L: {fmt(total_losses_val)}\n\n"
        f"Total MFE: {fmt(total_mfe)}\n"
        f"Total MAE: {fmt(-total_mae)}\n"
        f"MFE / MAE Ratio: {mfe_mae_ratio:.2f}"
    )
    props = dict(boxstyle='round', facecolor='#d0e8d0', alpha=0.9)
    ax1.text(1.02, 0.95, combined_stats_text, transform=ax1.transAxes,
             fontsize=13, verticalalignment='top', bbox=props, linespacing=1.5)

    # --- EV rolling subplot (ax_ev) ---
    ax_ev = fig.add_subplot(gs[1, :], sharex=ax1)
    
    rolling_ev_arr = np.array(rolling_ev)
    rolling_ev_clean = np.nan_to_num(rolling_ev_arr, nan=0.0)

    # Fill positive/negative
    ax_ev.fill_between(x_trades, 0, rolling_ev_clean, where=(rolling_ev_clean >= 0), facecolor='green', alpha=0.25)
    ax_ev.fill_between(x_trades, 0, rolling_ev_clean, where=(rolling_ev_clean < 0), facecolor='red', alpha=0.15)
    
    ax_ev.plot(x_trades, rolling_ev_clean, color='gray', linewidth=2, label=f'{rolling_trades}-Trade Rolling Avg')
    
    # Cum Avg Trade
    min_trade_for_avg = 10
    if len(x_trades) > min_trade_for_avg:
        ax_ev.plot(x_trades[min_trade_for_avg-1:], cum_avg[min_trade_for_avg-1:],
                   color='orange', linewidth=2, linestyle='--', label='Cum Avg Trade')

    ax_ev.axhline(0, color='gray', linewidth=0.5)
    ax_ev.set_ylabel(f'P/L ({unit_label})')
    ax_ev.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax_ev.legend(loc='upper left')

    # X Ticks
    rough_step = total_trades // 10 if total_trades > 10 else 1
    nice_steps = [10, 20, 50, 100, 200, 500, 1000]
    step = min(n for n in nice_steps if n >= rough_step)
    tick_positions = list(range(0, total_trades+1, step))
    ax_ev.set_xticks(tick_positions)
    ax_ev.set_xticklabels(tick_positions)
    ax_ev.set_xlabel('Trade #')

    # --- Middle: Daily cumulative (ax2) ---
    ax2 = fig.add_subplot(gs[2, :])
    x_daily = list(range(1, len(daily_stats)+1))

    ax2.plot(x_daily, daily_stats['Cumulative'], color='black', linewidth=2, label='Daily Cum P/L')
    ax2.plot(x_daily, daily_stats['Weekly_Cum'], color='red', linestyle='--', linewidth=1.5,
             label='Weekly Cum P/L', drawstyle='steps-post')

    # Monthly shading on daily chart
    unique_months_daily = daily_stats['MonthNum'].unique()
    for i, month in enumerate(unique_months_daily):
        if i % 2 == 0:
            month_mask = daily_stats['MonthNum'] == month
            if month_mask.any():
                start_idx = month_mask.idxmax() + 1 - 0.5
                end_idx = month_mask[::-1].idxmax() + 1 + 0.5
                ax2.axvspan(start_idx, end_idx, color='lightgrey', alpha=0.3, zorder=0)

    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.set_xticks(x_daily)
    ax2.set_xticklabels([d.strftime('%d-%m') for d in daily_stats['Date']], rotation=90)
    ax2.set_ylabel(f'Cumulative P/L ({unit_label})')
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.legend(loc='upper left')

    # --- Metrics (Sharpe, Sortino, ROMAD) ---
    cum_metrics = [0.0] + daily_stats['Cumulative'].tolist()
    
    # Max DD calculation
    running_max = -np.inf
    max_dd = 0.0
    for val in cum_metrics:
        if val > running_max: running_max = val
        dd = val - running_max
        if dd < max_dd: max_dd = dd
    
    romad = (total_net / abs(max_dd)) if max_dd != 0 else float(total_net)

    # Weekly ROMAD
    weekly_net = daily_stats.groupby('Week')['Net'].sum()
    weekly_cum = weekly_net.cumsum()
    running_max_week = -np.inf
    max_dd_week = 0.0
    for val in weekly_cum:
        running_max_week = max(running_max_week, val)
        dd = val - running_max_week
        if dd < max_dd_week: max_dd_week = dd
    w_romad = (weekly_cum.iloc[-1] / abs(max_dd_week)) if max_dd_week != 0 else float('inf')

    # Sharpe / Sortino
    daily_returns = daily_stats['Net']
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
    
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std()
    sortino_ratio = (daily_returns.mean() / downside_std) * np.sqrt(252) if downside_std != 0 else float('inf')

    cum_stats_text = (
        f"Peak: {fmt(max(cum_metrics) if cum_metrics else 0)}\n"
        f"Valley: {fmt(min(cum_metrics) if cum_metrics else 0)}\n"
        f"Max DD: {fmt(max_dd)}\n\n"
        f"D_ROMAD: {romad:.2f}\n"
        f"W_ROMAD: {w_romad:.2f}\n"
        f"Sharpe: {sharpe_ratio:.2f}\n"
        f"Sortino: {sortino_ratio:.2f}"
    )
    props2 = dict(boxstyle='round', facecolor='#d0e8d0', alpha=0.9)
    ax2.text(1.02, 0.95, cum_stats_text, transform=ax2.transAxes,
             fontsize=13, verticalalignment='top', bbox=props2, linespacing=1.5)

    # --- Bottom left: Daily Candles (ax3) ---
    ax3 = fig.add_subplot(gs[3, 0])
    for i, row in enumerate(daily_stats.itertuples(), start=1):
        open_val = 0.0
        close_val = row.Net
        high_val = max(open_val, row.Intraday_High if pd.notnull(row.Intraday_High) else close_val, close_val)
        low_val = min(open_val, row.Intraday_Low if pd.notnull(row.Intraday_Low) else close_val, close_val)
        ax3.vlines(i, low_val, high_val, color='black', linewidth=1, zorder=1)
    
    for i, net_val in enumerate(daily_stats['Net'], start=1):
        open_val = 0.0
        ax3.add_patch(plt.Rectangle(
            (i-0.2, min(open_val, net_val)),
            0.4,
            abs(net_val-open_val),
            facecolor='white' if net_val>=open_val else 'black',
            edgecolor='black',
            linewidth=1,
            zorder=2
        ))

    # Monthly shading (daily chart)
    for i, month in enumerate(unique_months_daily):
        if i % 2 == 0:
            month_mask = daily_stats['MonthNum'] == month
            if month_mask.any():
                start_idx = month_mask.idxmax() + 1 - 0.5
                end_idx = month_mask[::-1].idxmax() + 1 + 0.5
                ax3.axvspan(start_idx, end_idx, color='lightgrey', alpha=0.3, zorder=0)

    ax3.axhline(0, color='gray', linewidth=0.5)
    ax3.set_xticks(x_daily)
    ax3.set_xticklabels([d.strftime('%d-%m') for d in daily_stats['Date']], rotation=90)
    ax3.set_ylabel(f'P/L ({unit_label})')
    ax3.set_title('Daily Candlesticks (Closed P/L)')
    ax3.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # --- Bottom Right: Weekly Table (ax4) ---
    ax4 = fig.add_subplot(gs[3, 1])
    ax4.axis('off')

    df_plot['Month'] = df_plot['Entry DateTime'].dt.to_period('M')
    
    # Handle Commission if present, else 0
    commission_col = 'Commission (C)' if 'Commission (C)' in df_plot.columns else 'Commission'
    if commission_col not in df_plot.columns:
        df_plot[commission_col] = 0.0

    weekly_stats = df_plot.groupby(['Month', 'Week']).agg(
        Net=('Display_PnL', 'sum'),
        Commissions=(commission_col, 'sum'),
        Trades=('Display_PnL', 'count')
    ).reset_index()

    weekly_stats['MonthLabel'] = weekly_stats['Month'].dt.strftime('%b')
    weekly_stats['WeekLabel'] = 'W' + weekly_stats['Week'].astype(str) + ' (' + weekly_stats['MonthLabel'] + ')'
    
    monthly_totals = df_plot.groupby('Month')['Display_PnL'].sum()
    weekly_stats['Month_Net'] = ''
    for month, group in weekly_stats.groupby('Month'):
        last_idx = group.index[-1]
        weekly_stats.loc[last_idx, 'Month_Net'] = fmt(monthly_totals[month])

    table_rows = weekly_stats[['WeekLabel', 'Trades', 'Net', 'Commissions', 'Month_Net']].copy()
    
    # Format
    table_rows['Net'] = table_rows['Net'].apply(fmt)
    table_rows['Commissions'] = table_rows['Commissions'].apply(fmt)
    
    # Colors
    cell_colors = []
    for _, r in weekly_stats.iterrows(): # Iterate generic stats for logic
        row_colors = []
        net_val = r['Net']
        # Try to parse monthly total
        try:
             # simple look up based on if this row has text
             month_txt = table_rows.loc[_, 'Month_Net']
             if month_txt:
                 val = float(month_txt.replace(currency_symbol,'').replace(',',''))
                 month_negative = val < 0
             else:
                 month_negative = False
        except:
             month_negative = False
             
        for col_name in ['WeekLabel', 'Trades', 'Net', 'Commissions', 'Month_Net']:
            if col_name == 'Net' and net_val < 0:
                row_colors.append('#e0e0e0')
            elif col_name == 'Month_Net' and month_negative:
                row_colors.append('#e0e0e0')
            else:
                row_colors.append('white')
        cell_colors.append(row_colors)

    columns = ['Week', 'Trades', 'Week Net', 'Comms', 'Month Net']
    table = ax4.table(
        cellText=table_rows.values.tolist(),
        colLabels=columns,
        cellLoc='center',
        loc='center',
        bbox=[0.05, 0.05, 0.9, 0.9],
        cellColours=cell_colors
    )
    
    # Table Styling
    col_widths = [0.15, 0.15, 0.2, 0.2, 0.2]
    for i in range(len(table_rows)+1):
        for j, w in enumerate(col_widths):
            try: table[i, j].set_width(w)
            except: pass
    for j in range(len(columns)):
        table[0, j].set_facecolor('#d0e8d0')

    table.auto_set_font_size(True)
    table.scale(1, 1.2)

   # --- Daily Stats Box (top) ---
    win_days = daily_stats[daily_stats['Net'] > 0]
    loss_days = daily_stats[daily_stats['Net'] < 0]
    total_days = len(daily_stats)

    win_rate_days = len(win_days) / total_days * 100 if total_days > 0 else 0
    avg_win_day = win_days['Net'].mean() if not win_days.empty else 0
    avg_loss_day = abs(loss_days['Net'].mean()) if not loss_days.empty else 0
    rr_day = (avg_win_day / avg_loss_day) if avg_loss_day != 0 else 0

    total_net_day = daily_stats['Net'].sum() if total_days > 0 else 0
    avg_day = daily_stats['Net'].mean() if total_days > 0 else 0
    best_day = daily_stats['Net'].max() if not daily_stats.empty else 0
    worst_day = daily_stats['Net'].min() if not daily_stats.empty else 0

    daily_stats_text = (
        f"Total Days: {total_days}\n"
        f"Total Net: {fmt(total_net_day)}\n"
        f"Win% Days: {win_rate_days:.1f}%\n"
        f"Avg W / Avg L: {rr_day:.2f}\n"
        f"Avg Day: {fmt(avg_day)}\n"
        f"Avg W Day: {fmt(avg_win_day)}\n"
        f"Avg L Day: {fmt(-avg_loss_day)}\n"
        f"Best Day: {fmt(best_day)}\n"
        f"Worst Day: {fmt(worst_day)}"
    )

    props_daily = dict(boxstyle='round', facecolor='#d0e8d0', alpha=0.9)
    ax4.text(1.02, 0.95, daily_stats_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', bbox=props_daily)


    # --- Weekly Stats Box (bottom) ---
    win_weeks = weekly_stats[weekly_stats['Net'] > 0]
    loss_weeks = weekly_stats[weekly_stats['Net'] < 0]
    total_weeks = len(weekly_stats)

    win_rate_weeks = len(win_weeks) / total_weeks * 100 if total_weeks > 0 else 0
    avg_win_week = win_weeks['Net'].mean() if not win_weeks.empty else 0
    avg_loss_week = abs(loss_weeks['Net'].mean()) if not loss_weeks.empty else 0
    rr_week = (avg_win_week / avg_loss_week) if avg_loss_week != 0 else 0

    avg_week = weekly_stats['Net'].mean() if total_weeks > 0 else 0
    best_week = weekly_stats['Net'].max() if not weekly_stats.empty else 0
    worst_week = weekly_stats['Net'].min() if not weekly_stats.empty else 0

    weekly_stats_text = (
        f"Total Weeks: {total_weeks}\n"
        f"Win% Weeks: {win_rate_weeks:.1f}%\n"
        f"Avg W / Avg L: {rr_week:.2f}\n"
        f"Avg Week: {fmt(avg_week)}\n"
        f"Avg W Week: {fmt(avg_win_week)}\n"
        f"Avg L Week: {fmt(-avg_loss_week)}\n"
        f"Best Week: {fmt(best_week)}\n"
        f"Worst Week: {fmt(worst_week)}"
    )

    ax4.text(1.02, 0.45, weekly_stats_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', bbox=props_daily)


    # ---- FINAL LAYOUT ----
    plt.tight_layout(rect=[0, 0, 0.97, 1])
    
    # Close gap between ax1 and ax_ev
    pos1 = ax1.get_position()
    pos_ev = ax_ev.get_position()
    new_y0 = pos_ev.y0
    new_height = pos1.y0 - new_y0
    ax_ev.set_position([pos_ev.x0, new_y0, pos_ev.width, new_height])

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf