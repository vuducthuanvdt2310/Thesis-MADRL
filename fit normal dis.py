import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Tên file của bạn
file_path = 'Final Inputdata.xlsx'

def fit_and_plot_normal_distribution(df, sheet_name):
    print(f"--- KẾT QUẢ PHÂN PHỐI CHUẨN: TỜ '{sheet_name.upper()}' ---")
    
    # Lấy các cột bắt đầu bằng chữ 'SKU'
    sku_columns = [col for col in df.columns if str(col).startswith('SKU')]
    
    # Thiết lập khung vẽ biểu đồ
    fig, axes = plt.subplots(1, len(sku_columns), figsize=(15, 4))
    if len(sku_columns) == 1:
        axes = [axes]
        
    results = {}
    
    for i, sku in enumerate(sku_columns):
        # Chuyển đổi dữ liệu cột thành mảng (array) và loại bỏ các giá trị rỗng (NaN) nếu có
        data_array = df[sku].dropna().values
        
        # Fit dữ liệu với hàm Normal Distribution để tìm Mean và Std Dev
        mu, std = norm.fit(data_array)
        results[sku] = {'mean': mu, 'std': std}
        
        print(f"{sku}: Mean (μ) = {mu:.2f}, Std (σ) = {std:.2f}")
        
        # --- Phần vẽ biểu đồ trực quan ---
        ax = axes[i]
        # Vẽ Histogram của dữ liệu thực tế
        ax.hist(data_array, bins=10, density=True, alpha=0.6, color='b', edgecolor='black')
        
        # Vẽ đường cong phân phối chuẩn lý thuyết
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=2)
        
        ax.set_title(f"{sku} {sheet_name}\n$\mu={mu:.1f},\ \sigma={std:.1f}$")
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')

    plt.tight_layout()
    plt.show()
    
    return results

if __name__ == "__main__":
    try:
        # Đọc dữ liệu từ file Excel
        df_demand = pd.read_excel(file_path, sheet_name='demand')
        df_price = pd.read_excel(file_path, sheet_name='price')
        
        # Chạy hàm fit và vẽ biểu đồ cho sheet demand
        demand_params = fit_and_plot_normal_distribution(df_demand, 'demand')
        print("\n")
        
        # Chạy hàm fit và vẽ biểu đồ cho sheet price
        price_params = fit_and_plot_normal_distribution(df_price, 'price')
        
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{file_path}'. Vui lòng kiểm tra lại đường dẫn.")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")