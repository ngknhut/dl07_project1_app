# import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import squarify
import plotly.express as px
from sklearn.preprocessing import RobustScaler
import pickle
import joblib
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import io
from sklearn.metrics import silhouette_score
import datetime
from sklearn.cluster import KMeans

# Thiết lập cấu hình trang
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_product_df():
    try:
        df = pd.read_csv('Products_with_Categories.csv')
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file CSV: {e}")
        return pd.DataFrame()  # Trả về DataFrame trống nếu có lỗi
    
@st.cache_data
def load_transactions_df():
    try:
        df = pd.read_csv('Transactions.csv')
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file CSV: {e}")
        return pd.DataFrame()  # Trả về DataFrame trống nếu có lỗi

@st.cache_resource
def load_model(path='customer_segmentation_pipeline.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)
        
#pipeline = joblib.load('customer_segmentation_pipeline.pkl')
pipeline = load_model()
products_df = load_product_df()
transactions_df = load_transactions_df()
transactions_df['Date'] = pd.to_datetime(transactions_df['Date'], format='%d-%m-%Y')
transactions_df['order_id'] = transactions_df.groupby(['Member_number', 'Date']).ngroup() + 1
merged_df = pd.merge(transactions_df, products_df, on='productId', how='left')
merged_df['Total_Cost'] = merged_df['price'] * merged_df['items']
merged_df['Month'] = merged_df['Date'].dt.to_period('M')
monthly_category_transactions = merged_df.groupby(['Month', 'Category']).size().reset_index(name='Transaction_Count')
pivot_table = monthly_category_transactions.pivot(index='Category', columns='Month', values='Transaction_Count').fillna(0)
category_counts = products_df.groupby('Category')['productName'].nunique().reset_index(name='Product_Count')
category_counts = category_counts.sort_values(by='Product_Count', ascending=False)
top_products = merged_df.groupby('productName')['Total_Cost'].sum().reset_index()
top_10_products = top_products.sort_values(by='Total_Cost', ascending=False).head(10)
top_products = merged_df.groupby('productName')['items'].sum().sort_values(ascending=False).head(10)
top_categories = merged_df.groupby('Category')['Total_Cost'].sum().reset_index()
top_10_categories = top_categories.sort_values(by='Total_Cost', ascending=False).head(10)
fresh_food_df = merged_df[merged_df['Category'] == 'Fresh Food']
top_fresh_food = fresh_food_df.groupby('productName')['Total_Cost'].sum().reset_index()
top_10_fresh_food = top_fresh_food.sort_values(by='Total_Cost', ascending=False).head(10)

df = merged_df
current_date = df['Date'].max()
rfm_df = df.groupby('Member_number').agg({
'Date': lambda x: (current_date - x.max()).days,  # Recency: Số ngày từ giao dịch cuối cùng
'order_id': 'nunique',                            # Frequency: Số đơn hàng duy nhất
'Total_Cost': 'sum'                               # Monetary: Tổng chi tiêu
}).reset_index()
rfm_df.columns = ['Member_number', 'Recency', 'Frequency', 'Monetary']
rfm_df = rfm_df.sort_values('Monetary', ascending=False)
# Create labels for Recency, Frequency, Monetary
r_labels = range(4, 0, -1) # số ngày tính từ lần cuối mua hàng lớn thì gán nhãn nhỏ, ngược lại thì nhãn lớn
f_labels = range(1, 5)
m_labels = range(1, 5)
# Assign these labels to 4 equal percentile groups
r_groups = pd.qcut(rfm_df['Recency'].rank(method='first'), q=4, labels=r_labels)
f_groups = pd.qcut(rfm_df['Frequency'].rank(method='first'), q=4, labels=f_labels)
m_groups = pd.qcut(rfm_df['Monetary'].rank(method='first'), q=4, labels=m_labels)
rfm_df = rfm_df.assign(R = r_groups.values, F = f_groups.values,  M = m_groups.values)
def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
rfm_df['RFM_Segment'] = rfm_df.apply(join_rfm, axis=1)
rfm_count_unique = rfm_df.groupby('RFM_Segment')['RFM_Segment'].nunique()
rfm_df['RFM_Score'] = rfm_df[['R','F','M']].sum(axis=1)
def rfm_level(df):
    if df['RFM_Score'] == 12:
        return 'Best Customers'  # High recency, frequency, and monetary
    elif df['R'] == 1 and df['F'] == 1 and df['M'] == 1:
        return 'New Customers'  # Very low recency, frequency, and monetary
    elif df['M'] == 4:
        return 'Big Spenders'  # High monetary
    elif df['F'] == 4:
        return 'Loyal Customers'  # High frequency
    elif df['R'] == 4:
        return 'Active Customers'  # High recency
    else:
        return 'At-Risk/Occasional'  # All other cases
rfm_df['RFM_Level'] = rfm_df.apply(rfm_level, axis=1)
output_file = 'rfm_df.csv'
rfm_df.to_csv(output_file, index=True, encoding='utf-8')

def highlight_kmeans(val):
    if val == 'VIP':
        return 'background-color: #006400; color: white; font-weight: bold'
    elif val == 'Loyal Customers':
        return 'background-color: #006400; color: white; font-weight: bold'
    elif val == 'At-Risk Customers':
        return 'background-color: #FFA500; color: white; font-weight: bold'
    elif val == 'Lost Customers':
        return 'background-color: #8B0000; color: white; font-weight: bold'
    else:
        return ''

def highlight_RFM(val):
    if val == 'Active Customers':
        return 'background-color: #ffff00; color: white; font-weight: bold'
    elif val == 'Big Spenders':
        return 'background-color: #4169e1; color: white; font-weight: bold'
    elif val == 'Occasional Customers':
        return 'background-color: #00ffff; color: white; font-weight: bold'
    elif val == 'Lost Customers':
        return 'background-color: #ff0000; color: white; font-weight: bold'
    elif val == 'Loyal Customers':
        return 'background-color: #800080; color: white; font-weight: bold'
    elif val == 'New Customers':
        return 'background-color: #008000; color: white; font-weight: bold'
    elif val == 'Best Customers':
        return 'background-color: #ffd700; color: white; font-weight: bold'
    else:
        return ''

st.sidebar.title("Customer Segmentation")
st.sidebar.image("side_bar_img.jpg", width=250)
st.sidebar.markdown("---")

menu = ["Giới thiệu tổng quan", "EDA","Huấn luyện mô hình","Tra cứu nhóm khách hàng"]
choice = st.sidebar.selectbox('Menu', menu)

# Thiết lập thông tin trong sidebar
# Separator
st.sidebar.markdown('<div class="separator"></div>', unsafe_allow_html=True)   

st.sidebar.markdown("👨‍🏫 **Giảng viên:** Cô Khuất Thùy Phương")
# Separator
st.sidebar.markdown('<div class="separator"></div>', unsafe_allow_html=True)   


# Footer info
st.sidebar.markdown("🏆 **Thực hiện bởi:**")

st.sidebar.markdown("""
<div style="padding: 2px; border-radius: 5px; margin: 5px 0;">
    <p style="color: white; margin: 0;">Cao Thị Ngọc Minh & Nguyễn Kế Nhựt</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<style>
    .separator {
        margin: 5px 0;
        height: 1px;
        background-color: #444;
        width: 100%;
    }
</style>
<div class="separator"></div>
""", unsafe_allow_html=True)

today = datetime.date.today().strftime("%d/%m/%Y")
st.sidebar.markdown(f"📅 **Ngày báo cáo:** {today}")


if choice == 'Giới thiệu tổng quan':    
    st.image('rfm-1.jpg', caption='Hình ảnh giới thiệu về dự án', use_container_width=True) 
    # Thông tin tổng quan về dự án
    st.header("1. Giới thiệu về dự án",divider='gray')
    st.markdown("""
    <ul style='font-size: 18px;'>
        <li>Ứng dụng dành riêng cho **chủ cửa hàng X**</li>
        <li>Phân tích hành vi khách hàng dựa trên dữ liệu giao dịch</li>
        <li>Cung cấp công cụ trực quan, dễ sử dụng</li>
        <li>Hỗ trợ ra quyết định kinh doanh hiệu quả</li>
    </ul>
    """, unsafe_allow_html=True)

    st.header("2. Kết quả đạt được",divider='gray')
    st.markdown("""
    <ul style='font-size: 18px;'>
        <li>Xác định các phân nhóm khách hàng rõ ràng</li>
        <li>Dựa trên thói quen mua sắm và sở thích</li>
        <li>Phân tích mức độ chi tiêu của khách hàng</li>
        <li>Hỗ trợ xây dựng chiến lược tiếp cận phù hợp</li>
    </ul>
    """, unsafe_allow_html=True)

    st.header("3. Lợi ích cho người dùng",divider='gray')
    st.markdown("""
    <ul style='font-size: 18px;'>
        <li>Giao diện thân thiện, dễ thao tác</li>
        <li>Xác định khách hàng tiềm năng nhanh chóng</li>
        <li>Cá nhân hóa chiến dịch tiếp thị hiệu quả</li>
    </ul>
    """, unsafe_allow_html=True)

    st.header("4. Hướng dẫn sử dụng trang web",divider='gray')
    st.markdown("""
    <ul style='font-size: 18px;'>
        <li>Giới thiệu tổng quan: Mô tả dự án, kết quả, và lợi ích, bao gồm hướng dẫn sử dụng</li>
        <li>EDA: Phân tích dữ liệu giao dịch qua bảng, trực quan hóa qua các biểu đồ</li>
        <li>Huấn luyện mô hình: Kết quả huấn luyện mô hình với 2 thuật toán: Manual RFM & KMean clustering  </li>
        <li>Tra cứu nhóm khách hàng: Dự đoán nhóm khách hàng dựa trên RFM, hỗ trợ nhập tay hoặc file</li>
    </ul>
    """, unsafe_allow_html=True)

elif choice == 'EDA':

    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(10, 6))  # Tạo figure và axes với kích thước 10x6
    sns.histplot(data=products_df, x='price', bins=20, kde=True, ax=ax)  # Vẽ histogram với KDE
    # Đặt tiêu đề và nhãn
    ax.set_title('Phân bố giá sản phẩm')
    ax.set_xlabel('Giá (Price)')
    ax.set_ylabel('Số lượng')
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

    # Tạo biểu đồ
    fig1, ax1 = plt.subplots(figsize=(12, 6))  # Tạo figure và axes với kích thước 12x6
    sns.barplot(data=category_counts, x='Category', y='Product_Count', ax=ax1)#palette='viridis'
    # Tùy chỉnh biểu đồ
    ax1.set_title('Số lượng sản phẩm theo danh mục', fontsize=14)
    ax1.set_xlabel('Danh mục (Category)', fontsize=12)
    ax1.set_ylabel('Số lượng sản phẩm (Product_Count)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45, labelright=False, labelleft=True)  # Xoay nhãn trục x 45 độ
    plt.tight_layout()  # Đảm bảo bố cục gọn gàng
    # Thêm số liệu trên mỗi cột
    for i, v in enumerate(category_counts['Product_Count']):
        ax1.text(i, v + 0.5, str(v), ha='center', va='bottom')
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig1)

    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(14, 8))  # Tạo figure và axes với kích thước 14x8
    sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlGnBu', cbar_kws={'label': 'Số lượng giao dịch'}, ax=ax)
    # Tùy chỉnh biểu đồ
    ax.set_title('Số lượng giao dịch theo thời gian và danh mục (Heatmap)')
    ax.set_xlabel('Tháng')
    ax.set_ylabel('Danh mục')
    ax.tick_params(axis='x', rotation=45)  # Xoay nhãn trục x 45 độ
    plt.tight_layout()  # Đảm bảo bố cục gọn gàng
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(10, 6))  # Tạo figure và axes với kích thước 10x6
    sns.histplot(data=merged_df, x='Total_Cost', bins=20, kde=True, ax=ax)
    # Tùy chỉnh biểu đồ
    ax.set_title('Phân bố tổng chi phí mỗi giao dịch')
    ax.set_xlabel('Tổng chi phí (Total Cost)')
    ax.set_ylabel('Số lượng')
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(12, 6))  # Tạo figure và axes với kích thước 12x6
    sns.barplot(data=top_10_products, x='productName', y='Total_Cost', ax=ax)#palette='viridis'
    # Tùy chỉnh biểu đồ
    ax.set_title('Top 10 mặt hàng có giá trị Total_Cost cao nhất', fontsize=14)
    ax.set_xlabel('Tên mặt hàng', fontsize=12)
    ax.set_ylabel('Doanh thu (Total_Cost)', fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelright=False, labelleft=True)  # Xoay nhãn trục x 45 độ
    plt.tight_layout()  # Đảm bảo bố cục gọn gàng
    # Thêm chú thích số trên mỗi cột
    for i, v in enumerate(top_10_products['Total_Cost']):
        ax.text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom')
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(12, 6))  # Tạo figure và axes với kích thước 12x6
    top_products.plot(kind='bar', ax=ax)  # Vẽ biểu đồ cột từ top_products
    # Tùy chỉnh biểu đồ
    ax.set_title('Top 10 sản phẩm được mua nhiều nhất')
    ax.set_xlabel('Tên sản phẩm')
    ax.set_ylabel('Tổng số lượng mua')
    ax.tick_params(axis='x', rotation=45)  # Xoay nhãn trục x 45 độ
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(12, 6))  # Tạo figure và axes với kích thước 12x6
    sns.countplot(y='Category', data=merged_df, order=merged_df['Category'].value_counts().index, ax=ax)
    # Tùy chỉnh biểu đồ
    ax.set_title('Số lượng giao dịch theo danh mục')
    ax.set_xlabel('Số lượng giao dịch')
    ax.set_ylabel('Danh mục')
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(12, 6))  # Tạo figure và axes với kích thước 12x6
    sns.barplot(data=top_10_categories, y='Category', x='Total_Cost', ax=ax)#palette='viridis'
    # Tùy chỉnh biểu đồ
    ax.set_title('Top 10 danh mục có doanh thu cao nhất', fontsize=14)
    ax.set_xlabel('Tổng doanh thu (Total_Cost)', fontsize=12)
    ax.set_ylabel('Danh mục (Category)', fontsize=12)
    plt.tight_layout()  # Đảm bảo bố cục gọn gàng
    # Thêm chú thích số trên mỗi cột
    for i, v in enumerate(top_10_categories['Total_Cost']):
        ax.text(v + 0.5, i, f'{v:.2f}', va='center')
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(12, 6))  # Tạo figure và axes với kích thước 12x6
    sns.barplot(data=top_10_fresh_food, y='productName', x='Total_Cost' , ax=ax)#palette='viridis'
    # Tùy chỉnh biểu đồ
    ax.set_title('Top 10 mặt hàng trong danh mục Fresh Food có doanh thu cao nhất', fontsize=14)
    ax.set_xlabel('Tổng doanh thu (Total_Cost)', fontsize=12)
    ax.set_ylabel('Tên mặt hàng (productName)', fontsize=12)
    plt.tight_layout()  # Đảm bảo bố cục gọn gàng
    # Thêm chú thích số trên mỗi cột
    for i, v in enumerate(top_10_fresh_food['Total_Cost']):
        ax.text(v + 0.5, i, f'{v:.2f}', va='center')
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)
    
elif choice=='Huấn luyện mô hình':
    st.subheader("Manual RFM")
        # Calculate average values for each RFM_Level, and return a size of each segment
    rfm_agg = rfm_df.groupby('RFM_Level').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']}).round(0)

    rfm_agg.columns = rfm_agg.columns.droplevel()
    rfm_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
    rfm_agg['Percent'] = round((rfm_agg['Count']/rfm_agg.Count.sum())*100, 2)
    # Reset the index
    rfm_agg = rfm_agg.reset_index()

    # Tạo figure với kích thước tổng thể
    fig, axes = plt.subplots(3, 1, figsize=(12, 6))  # 3 hàng, 1 cột, kích thước 12x6
    # Vẽ phân phối của 'Recency'
    axes[0].hist(rfm_df['Recency'], bins=20, edgecolor='black')  # Histogram với 20 bins
    axes[0].set_title('Distribution of Recency')
    axes[0].set_xlabel('Recency')
    # Vẽ phân phối của 'Frequency'
    axes[1].hist(rfm_df['Frequency'], bins=10, edgecolor='black')  # Histogram với 10 bins
    axes[1].set_title('Distribution of Frequency')
    axes[1].set_xlabel('Frequency')
    # Vẽ phân phối của 'Monetary'
    axes[2].hist(rfm_df['Monetary'], bins=20, edgecolor='black')  # Histogram với 20 bins
    axes[2].set_title('Distribution of Monetary')
    axes[2].set_xlabel('Monetary')
    # Đảm bảo bố cục gọn gàng
    plt.tight_layout()
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)
    st.table(rfm_df['RFM_Level'].value_counts())
    st.table(rfm_agg)

    # Định nghĩa từ điển màu sắc
    colors_dict = {
        'Active Customers': 'yellow',
        'Big Spenders': 'royalblue',
        'Occasional Customers': 'cyan',
        'Lost Customers': 'red',
        'Loyal Customers': 'purple',
        'New Customers': 'green',
        'Best Customers': 'gold'
    }
    # Tạo figure và axes
    fig = plt.figure()  # Tạo figure
    ax = fig.add_subplot()  # Thêm subplot
    fig.set_size_inches(14, 10)  # Đặt kích thước 14x10
    # Vẽ treemap
    squarify.plot(
        sizes=rfm_agg['Count'],  # Kích thước ô dựa trên số lượng khách hàng
        text_kwargs={'fontsize': 12, 'weight': 'bold', 'fontname': 'sans serif'},  # Tùy chỉnh văn bản
        color=colors_dict.values(),  # Gán màu từ từ điển
        label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg.iloc[i])
                      for i in range(0, len(rfm_agg))],  # Nhãn với thông tin chi tiết
        alpha=0.5  # Độ trong suốt
    )
    # Tùy chỉnh biểu đồ
    plt.title("Customers Segments_Manual RFM", fontsize=26, fontweight="bold")
    plt.axis('off')  # Tắt trục
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

    # Tạo biểu đồ phân tán
    fig = px.scatter(
        rfm_agg,
        x="RecencyMean",
        y="MonetaryMean",
        size="FrequencyMean",
        color="RFM_Level",
        hover_name="RFM_Level",
        size_max=100
    )
    # Hiển thị biểu đồ trong Streamlit
    st.plotly_chart(fig, use_container_width=True)
  
    st.subheader("Kmeans RFM")
    
    rfm_df= pd.read_csv('rfm_df.csv')
    df_now = rfm_df[['Recency','Frequency','Monetary']]
    rfm_df['Log_Recency'] = np.log1p(rfm_df['Recency'])
    rfm_df['Log_Frequency'] = np.log1p(rfm_df['Frequency'])
    rfm_df['Log_Monetary'] = np.log1p(rfm_df['Monetary'])
    scaler = RobustScaler()
    rfm_df[['Scaled_Log_Recency', 'Scaled_Log_Frequency', 'Scaled_Log_Monetary']] = scaler.fit_transform(
        rfm_df[['Log_Recency', 'Log_Frequency', 'Log_Monetary']])
    # Elbow Method để chọn k
    X = rfm_df[['Scaled_Log_Recency', 'Scaled_Log_Frequency', 'Scaled_Log_Monetary']]

    range_n_clusters = range(2, 11)

    # Tính toán Silhouette Score và SSE
    silhouette_avg_list = []
    sse_list = []

    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Silhouette Score
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_avg_list.append(silhouette_avg)
        
        # SSE
        sse = kmeans.inertia_
        sse_list.append(sse)

    # Tính SSE%
    sse_percent_drop = [0]  # Phần trăm giảm đầu tiên = 0
    for i in range(1, len(sse_list)):
        drop = ((sse_list[i-1] - sse_list[i]) / sse_list[i-1]) * 100
        sse_percent_drop.append(drop)
        
    # Lấy các cột đã chuẩn hóa từ rfm_df
    df_now_scaled = rfm_df[['Scaled_Log_Recency', 'Scaled_Log_Frequency', 'Scaled_Log_Monetary']]
    # Thực hiện phân cụm với k = 4
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(df_now_scaled)
    # Gán nhãn phân cụm vào cột 'Cluster' trong rfm_df
    rfm_df['Cluster'] = kmeans.labels_
    # Tính trung bình và đếm số lượng cho từng cụm
    rfm_agg2 = rfm_df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']
    }).round(0)
    # Phương pháp 2: Đổi tên các mục trong chú thích
    cluster_names = {
        0: 'Loyal Customers',
        1: 'At-Risk Customers',
        2: 'VIP',
        3: 'Lost Customers'
    }
    # Đổi tên cột
    rfm_agg2.columns = rfm_agg2.columns.droplevel()
    rfm_agg2.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
    # Tính phần trăm
    rfm_agg2['Percent'] = round((rfm_agg2['Count'] / rfm_agg2.Count.sum()) * 100, 2)
    rfm_agg2 = rfm_agg2.reset_index()

    # Đổi kiểu dữ liệu cột Cluster
    rfm_agg2['Cluster'] = 'Cluster ' + rfm_agg2['Cluster'].astype('str')

    # Vẽ biểu đồ
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Biểu đồ Silhouette
    axes[0].plot(range_n_clusters, silhouette_avg_list, 'o-', color='blue')
    axes[0].set_title('Silhouette Score vs K')
    axes[0].set_xlabel('Số cụm (K)')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].grid(True)
    axes[0].set_xticks(list(range_n_clusters))

    # Biểu đồ SSE%
    axes[1].plot(range_n_clusters, sse_percent_drop, 'o-', color='green')
    axes[1].set_title('Tỷ lệ giảm SSE theo K (SSE%)')
    axes[1].set_xlabel('Số cụm (K)')
    axes[1].set_ylabel('Giảm SSE so với K trước (%)')
    axes[1].grid(True)
    axes[1].set_xticks(list(range_n_clusters))

    plt.suptitle('Đánh giá số lượng cụm tối ưu (K) theo Silhouette & SSE%', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

    # Tạo ánh xạ từ giá trị Cluster sang tên có ý nghĩa
    cluster_mapping = {
        'Cluster 0': 'Loyal Customers',
        'Cluster 1': 'At-Risk Customers',
        'Cluster 2': 'VIP',
        'Cluster 3': 'Lost Customers'
    }

    # Tạo cột mới trong DataFrame
    rfm_agg2['ClusterName'] = rfm_agg2['Cluster'].map(cluster_mapping)
    # Tạo biểu đồ phân tán
    fig = px.scatter(
        rfm_agg2,
        x="RecencyMean",
        y="MonetaryMean",
        size="FrequencyMean",
        color="ClusterName",
        hover_name="ClusterName",
        size_max=100
    )
    
    # Hiển thị biểu đồ trong Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Định nghĩa từ điển màu sắc
    colors_dict2 = {
        'Cluster0': 'yellow',
        'Cluster1': 'royalblue',
        'Cluster2': 'cyan',
        'Cluster3': 'red',
    }

        # Định nghĩa từ điển ánh xạ
    cluster_to_group = {
        0: 'Loyal Customers',
        1: 'At-Risk Customers',
        2: 'VIP',
        3: 'Lost Customers'
    }

    # Thêm cột 'Cluster_Name' vào rfm_agg2
    rfm_agg2['Cluster_Name'] = rfm_agg2['Cluster'].apply(lambda x: cluster_to_group[int(x.split()[-1])])

    # Tạo figure và axes
    fig = plt.figure()
    ax = fig.add_subplot()
    fig.set_size_inches(14, 10)

    # Vẽ Treemap
    squarify.plot(
        sizes=rfm_agg2['Count'],
        text_kwargs={'fontsize': 12, 'weight': 'bold', 'fontname': "sans serif"},
        color=colors_dict2.values(),
        label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(
            rfm_agg2['Cluster_Name'].iloc[i],  # Sử dụng Cluster_Name thay vì Cluster
            rfm_agg2['RecencyMean'].iloc[i],
            rfm_agg2['FrequencyMean'].iloc[i],
            rfm_agg2['MonetaryMean'].iloc[i],
            rfm_agg2['Count'].iloc[i],
            rfm_agg2['Percent'].iloc[i]
        ) for i in range(0, len(rfm_agg2))],
        alpha=0.5
    )

    # Tùy chỉnh biểu đồ
    plt.title("Customers Segments_Kmeans RFM", fontsize=26, fontweight="bold")
    plt.axis('off')

    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)


elif choice=='Tra cứu nhóm khách hàng':
    
    # CSS để tùy chỉnh kích thước chữ cho badge
    st.markdown("""
    <style>
        /* Tùy chỉnh kích thước chữ cho badge */
        .highlight-label {
            background-color: #006400;
            width: 220px;
            color: white;
            padding: 5px 8px;
            border-radius: 5px;
            font-weight: bold;
            display: inline-block;
            margin: 5px 0;
            font-size: 18px; /* Thay đổi kích thước chữ ở đây */
        }
        
        /* Tạo thêm các kích thước khác nhau */
        .highlight-label.small {
            font-size: 14px;
        }
        
        .highlight-label.medium {
            font-size: 18px;
        }
        
        .highlight-label.large {
            font-size: 22px;
        }
        
        .highlight-label.x-large {
            font-size: 26px;
        }
        .highlight-label.fixed-width {
        width: 300px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Tạo CSS tùy chỉnh
    st.markdown("""
    <style>
        /* Tùy chỉnh cho tiêu đề cột */
        .dataframe thead th {
            background-color: #2C3E50 !important;
            color: white !important;
            font-weight: bold !important;
            text-align: center !important;
            padding: 10px !important;
            border-bottom: 2px solid #34495E !important;
        }
        
        /* Tùy chỉnh cho toàn bộ bảng */
        .dataframe {
            width: 100% !important;
            border-collapse: collapse !important;
        }
        
        .dataframe tbody tr {
            border-bottom: 1px solid #34495E !important;
        }
        
        .dataframe tbody tr:hover {
            background-color: rgba(52, 73, 94, 0.1) !important;
        }
    </style>
    """, unsafe_allow_html=True)

    cluster_to_group = {
        0: 'Loyal Customers', #Mua gần đây, tần suất ổn định, chi tiêu khá cao.
        1: 'At-Risk Customers', #Lâu không mua, tần suất thấp, chi tiêu không nổi bật.
        2: 'VIP', #Mua thường xuyên, chi tiêu cao, dù không phải gần đây nhất.
        3: 'Lost Customers' #Lâu không mua, hiếm khi mua, chi tiêu rất thấp.
    }
    # Chọn nhập mã khách hàng hoặc nhập thông tin khách hàng vào dataframe
    #st.subheader("Chọn cách nhập thông tin khách hàng",divider='gray')
    st.markdown('<h2 class="main-title">📊 Hệ thống phân nhóm khách hàng</h1>', unsafe_allow_html=True)
    # CSS tùy chỉnh để thay đổi kích thước chữ của tab
    st.markdown("""
    <style>
        /* Điều chỉnh kích thước chữ trong tab với !important để ghi đè */
        .stTabs [data-baseweb="tab"] {
            font-size: 16px !important;  /* Tăng kích thước chữ lên đáng kể */
            font-weight: 500 !important; /* Độ đậm của chữ */
        }
        
        /* Sử dụng selector cụ thể hơn và !important để tăng độ ưu tiên */
        div[data-testid="stVerticalBlock"] .stTabs [data-baseweb="tab"] span {
            font-size: 16px !important;
        }
        
        /* Thêm CSS cho văn bản bên trong tab */
        .stTabs [data-baseweb="tab"] > div {
            font-size: 16px !important;
        }
        
        /* Tùy chỉnh thêm cho tab đang được chọn */
        .stTabs [aria-selected="true"] {
            font-weight: 600 !important; /* Chữ đậm hơn cho tab đang được chọn */
        }
        
        /* CSS cho màn hình kiểm tra kích thước chữ */
        .font-size-demo {
            border: 1px solid #ddd;
            padding: 10px;
            margin-top: 20px;
            background-color: #f9f9f9;
        }
        
        .font-size-demo p {
            margin: 5px 0;
        }
        
        .font-size-12 { font-size: 12px; }
        .font-size-14 { font-size: 14px; }
        .font-size-16 { font-size: 16px; }
        .font-size-18 { font-size: 18px; }
        .font-size-20 { font-size: 20px; }
        .font-size-24 { font-size: 24px; }
    </style>
    """, unsafe_allow_html=True)
    tabs = st.tabs(["Nhập mã khách hàng", "Nhập thông tin trực tiếp", "Tải file Excel/CSV"])

    with tabs[0]:
        st.markdown('<span class="highlight-label medium">1. Nhập mã khách hàng</span>', unsafe_allow_html=True)
        customer_id = st.text_input("Nhập mã khách hàng",placeholder="Nhập mã khách hàng",label_visibility="collapsed")
        # Nếu người dùng nhập mã khách hàng, thực hiện các xử lý tiếp theo
        # Đề xuất khách hàng thuộc cụm nào
        # In kết quả ra màn hình
        
        if customer_id:  # Kiểm tra nếu có mã khách hàng
            try:
                
                customer_data = rfm_df[rfm_df['Member_number'] == int(customer_id)][['Recency', 'Frequency', 'Monetary','RFM_Level']]
                styled_df_ = customer_data.style.map(
                    highlight_RFM, 
                    subset=['RFM_Level']
                )
                if not customer_data.empty:
                    st.write("Mã khách hàng:", customer_id)
                    st.markdown('<span class="highlight-label medium">2. Thông tin RFM</span>', unsafe_allow_html=True)
                    #st.badge("Thông tin RFM",color='green')

                    st.dataframe(styled_df_,hide_index=True)
                    cluster = pipeline.predict(customer_data[['Recency', 'Frequency', 'Monetary']])
                    group_name = cluster_to_group[cluster[0]]
                    st.markdown('<span class="highlight-label medium">3. Phân cụm khách hàng</span>', unsafe_allow_html=True)
                    st.badge(f"Khách hàng thuộc cụm theo Kmeans_RFM: {group_name}",color='orange')
                    rfm_level = customer_data['RFM_Level'].iloc[0]
                    st.badge(f"Khách hàng thuộc cụm theo tập luận RFM: {rfm_level}",color='orange')
                else:
                    st.write("Không tìm thấy khách hàng với mã này.")
            except ValueError:
                st.write("Vui lòng nhập mã khách hàng hợp lệ (số nguyên).")
    #elif type == "Nhập thông tin khách hàng vào dataframe":
    with tabs[1]:
        # Nếu người dùng chọn nhập thông tin khách hàng vào dataframe có 3 cột là Recency, Frequency, Monetary
        st.markdown('<span class="highlight-label medium">1. Thông tin khách hàng</span>', unsafe_allow_html=True)

        customer_data = []
        for i in range(5):
            #st.write(f"Khách hàng {i+1}")
            st.badge(f"* Khách hàng {i+1}",color='orange')
            recency = st.slider("Recency (ngày)", 1, 365, 100, key=f"recency_{i}")
            frequency = st.slider("Frequency (đơn hàng)", 1, 50, 5, key=f"frequency_{i}")
            monetary = st.slider("Monetary ($)", 1, 1000, 100, key=f"monetary_{i}")
            customer_data.append({"Recency": recency, "Frequency": frequency, "Monetary": monetary})

        # Chuyển danh sách thành DataFrame
        df_customer = pd.DataFrame(customer_data)          
        # Thực hiện phân cụm khách hàng dựa trên giá trị của 3 cột này
        if not df_customer.empty:
            # Dự đoán cụm bằng pipeline
            clusters = pipeline.predict(df_customer)
            df_customer['Kmeans_RFM'] = [cluster_to_group[cluster] for cluster in clusters]
            styled_df = df_customer.style.map(
                highlight_kmeans, 
                subset=['Kmeans_RFM']
            )
        # In kết quả ra màn hình
        #st.write("##### 2. Phân cụm khách hàng")
        st.markdown('<span class="highlight-label medium">2. Phân cụm khách hàng</span>', unsafe_allow_html=True)
        #st.write(df_customer)
        st.dataframe(styled_df, use_container_width=True)

    with tabs[2]:
        # Nếu người dùng chọn tải file Excel/CSV
        #st.write("##### 1. Tải file template Excel hoặc CSV")
        st.markdown('<span class="highlight-label medium fixed-width">1. Tải file template Excel hoặc CSV</span>', unsafe_allow_html=True)
        # Tạo file mẫu để tải về
        sample_df = pd.DataFrame(columns=['Member_number', 'Recency', 'Frequency', 'Monetary'])
        # Chuyển DataFrame thành CSV buffer
        csv_buffer = io.StringIO()
        sample_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        # Chuyển DataFrame thành Excel buffer
        excel_buffer = io.BytesIO()
        sample_df.to_excel(excel_buffer, index=False)
        excel_data = excel_buffer.getvalue()
        # Thêm nút tải file mẫu (CSV và Excel)
        st.write("Tải file mẫu để điền dữ liệu:")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Tải file CSV mẫu",
                data=csv_data,
                file_name="customer_data_template.csv",
                mime="text/csv"
            )
        with col2:
            st.download_button(
                label="Tải file Excel mẫu",
                data=excel_data,
                file_name="customer_data_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )   
        # Upload file
        uploaded_file = st.file_uploader("Chọn file Excel hoặc CSV", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                # Đọc file dựa trên định dạng
                if uploaded_file.name.endswith('.csv'):
                    df_uploaded = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df_uploaded = pd.read_excel(uploaded_file)
                # Hiển thị dữ liệu ban đầu
                #st.write("##### 2. Dữ liệu từ file tải lên")
                st.markdown('<span class="highlight-label medium fixed-width">2. Dữ liệu từ file tải lên</span>', unsafe_allow_html=True)
                st.write(df_uploaded)
                # Kiểm tra các cột cần thiết
                required_columns = ['Recency', 'Frequency', 'Monetary']
                if all(col in df_uploaded.columns for col in required_columns):
                    # Dự đoán cụm
                    clusters = pipeline.predict(df_uploaded[required_columns])
                    df_uploaded['Kmeans_RFM'] = [cluster_to_group[cluster] for cluster in clusters]
                    # Hiển thị kết quả phân cụm
                    #st.write("##### 3. Kết quả phân cụm khách hàng")
                    st.markdown('<span class="highlight-label medium fixed-width">3. Phân cụm khách hàng</span>', unsafe_allow_html=True)
                    styled_df_1 = df_uploaded.style.map(
                        highlight_kmeans, 
                        subset=['Kmeans_RFM']
                    )
                    #st.write(df_uploaded)
                    st.dataframe(styled_df_1, use_container_width=True)
                else:
                    st.error("File tải lên cần có các cột: Recency, Frequency, Monetary")
            except Exception as e:
                st.error(f"Đã xảy ra lỗi khi xử lý file: {str(e)}")
        else:
            st.info("Vui lòng tải lên một file Excel hoặc CSV để bắt đầu.")
