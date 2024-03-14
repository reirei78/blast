import streamlit as st

# ドラッグアンドドロップをシミュレートする関数
def simulate_drag_and_drop():
    st.markdown("#### 画像ファイルをドラッグアンドドロップしてください")
    # ファイルがドラッグされたときの挙動をエミュレートする空のdiv要素
    st.markdown('<div id="drop_zone" style="border:2px dashed #aaa; padding:20px; text-align:center;"></div>', 
                unsafe_allow_html=True)

# メインのStreamlitアプリケーション
def main():
    st.title("画像のアップロード")

    # ドラッグアンドドロップをシミュレート
    simulate_drag_and_drop()

    # ファイルアップロードの処理
    uploaded_file = st.file_uploader("ファイルを選択してください", type=["jpg", "jpeg", "png"])

    # アップロードされた画像を表示
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

if __name__ == "__main__":
    main()