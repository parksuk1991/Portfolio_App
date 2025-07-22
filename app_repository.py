# -*- coding: utf-8 -*-
"""APP_repository.ipynb
"""

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import pearsonr
import requests
from PIL import Image
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="Portfolio Backtesting App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 벤치마크 이름 매핑
BENCHMARK_NAMES = {
    'SPY': 'S&P 500 지수',
    'QQQ': 'Nasdaq 100 지수',
    'ACWI': 'MSCI ACWI 지수'
}

# 벤치마크 옵션 (표시용)
BENCHMARK_OPTIONS = {
    'S&P 500 지수': 'SPY',
    'Nasdaq 100 지수': 'QQQ',
    'MSCI ACWI 지수': 'ACWI'
}

# 대체 자산 딕셔너리 (각 티커별 대체 후보 리스트)
ASSET_SUBSTITUTE_POOL = {
    'XLC': ['XTL', 'IYZ', 'VNQ'],
    'XLY': ['RTH', 'XRT', 'VCR'],
    'XLP': ['VDC', 'PBJ', 'SZK'],
    'XLE': ['VDE', 'IYE', 'DIG'],
    'XLF': ['VFH', 'IYF', 'KBE'],
    'XLV': ['VHT', 'IYH', 'PJP'],
    'XLI': ['VIS', 'IYJ', 'PPA'],
    'XLB': ['VAW', 'IYM', 'SLX'],
    'XLK': ['VGT', 'IYW', 'QQQ'],
    'XLU': ['VPU', 'IDU', 'PUI'],
    'SPYV': ['IVE', 'VTV', 'DVY'],
    'SPYG': ['IVW', 'VUG', 'MGK'],
    'VYM': ['DVY', 'VTV', 'SCHD'],
    'RSP': ['EQL', 'EWRS', 'SPY'],
    'USMV': ['SPLV', 'EFAV', 'SPY'],
    'SPMO': ['MTUM', 'PDP', 'QQQ'],
    'IDEV': ['EFA', 'VEA', 'ACWX'],
    'IEMG': ['EEM', 'VWO', 'SCHE'],
    'SPY': ['VOO', 'IVV', 'VTI'],
    'QQQ': ['VGT', 'XLK', 'MGK'],
    'ACWI': ['IXUS', 'VEA', 'EFA'],
    # 필요시 추가
}

def find_best_substitute_by_corr(target_ticker, start_date, end_date):
    """대체자산 후보군에서 데이터 길이가 더 길고 상관관계가 가장 높은 자산을 선택.
       Asset_Substitute_pool에 없거나 조건에 맞는 자산이 없으면 SPY 사용."""
    try:
        target_data = yf.download(target_ticker, start=start_date, end=end_date, progress=False)['Close'].dropna()
    except Exception:
        target_data = None

    substitutes = ASSET_SUBSTITUTE_POOL.get(target_ticker, [])
    best_corr = -2
    best_candidate = None
    best_candidate_data = None

    for substitute in substitutes:
        try:
            sub_data = yf.download(substitute, start=start_date, end=end_date, progress=False)['Close'].dropna()
            if target_data is not None and len(sub_data) <= len(target_data):
                continue

            merged = pd.concat([target_data, sub_data], axis=1, join='inner').dropna()
            if len(merged) < 12:
                continue
            returns1 = merged.iloc[:, 0].pct_change().dropna()
            returns2 = merged.iloc[:, 1].pct_change().dropna()
            common = returns1.index.intersection(returns2.index)
            if len(common) < 10:
                continue
            corr, _ = pearsonr(returns1.loc[common], returns2.loc[common])
            if np.isnan(corr):
                continue
            if corr > best_corr:
                best_corr = corr
                best_candidate = substitute
                best_candidate_data = sub_data
        except Exception:
            continue

    if best_candidate is not None:
        return best_candidate, best_candidate_data

    # pool에 없거나 조건맞는 후보가 없는 경우 SPY로 대체
    st.warning(f"❌ {target_ticker}: 적절한 대체 자산을 찾을 수 없어 'SPY' 데이터로 대체합니다.")
    try:
        spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)['Close'].dropna()
        return 'SPY', spy_data
    except Exception:
        return None, None

def fill_missing_data(tickers, start_date, end_date, fill_gaps=True):
    st.info("📊 데이터 로딩 및 공백 분석 중...")

    original_data = {}
    missing_tickers = []
    data_info = {}

    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)['Close']
            if isinstance(data, pd.Series):
                data = data.to_frame(name=ticker)
            data_start = data.first_valid_index()
            data_end = data.last_valid_index()
            data_length = len(data.dropna())
            target_start = pd.to_datetime(start_date)

            if data_start is None or data_length < 50:
                missing_tickers.append(ticker)
                st.warning(f"❌ {ticker}: 데이터 부족 (길이: {data_length})")
            elif data_start > target_start + pd.DateOffset(years=1):
                missing_tickers.append(ticker)
                st.warning(f"⚠️ {ticker}: 시작일 부족 (목표: {target_start.strftime('%Y-%m')}, 실제: {data_start.strftime('%Y-%m')})")
                data_info[ticker] = {
                    'original_data': data,
                    'start_gap': (data_start - target_start).days,
                    'needs_filling': True
                }
            else:
                original_data[ticker] = data
                data_info[ticker] = {
                    'original_data': data,
                    'start_gap': 0,
                    'needs_filling': False
                }
                st.success(f"✅ {ticker}: 데이터 양호 ({data_start.strftime('%Y-%m')} ~ {data_end.strftime('%Y-%m')})")
        except Exception as e:
            missing_tickers.append(ticker)
            st.error(f"❌ {ticker}: 데이터 로드 실패 - {str(e)}")

    if not fill_gaps or len(missing_tickers) == 0:
        if len(original_data) > 0:
            combined_data = pd.concat(original_data.values(), axis=1)
            combined_data.columns = original_data.keys()
            return combined_data.resample('ME').last().dropna(), {}
        else:
            return None, {}

    st.info("🔄 대체 자산 검색 및 데이터 결합 중...")

    substitution_log = {}
    enhanced_data = original_data.copy()

    for ticker in missing_tickers:
        substitute_ticker, substitute_data = find_best_substitute_by_corr(
            ticker, start_date, end_date
        )
        if substitute_ticker and substitute_data is not None:
            if isinstance(substitute_data, pd.Series):
                substitute_data = substitute_data.to_frame(name=substitute_ticker)
            substitute_df = substitute_data.copy()
            substitute_df.columns = [ticker]
            enhanced_data[ticker] = substitute_df
            substitution_log[ticker] = {
                'substitute': substitute_ticker,
                'original_start': data_info.get(ticker, {}).get('original_data', pd.DataFrame()).first_valid_index(),
                'substitute_start': substitute_data.first_valid_index(),
                'method': 'by_corr'
            }
        else:
            st.error(f"❌ {ticker}: 적절한 대체 자산을 찾을 수 없습니다.")

    if len(enhanced_data) > 0:
        final_data = pd.concat(enhanced_data.values(), axis=1)
        final_data.columns = enhanced_data.keys()
        monthly_data = final_data.resample('ME').last().dropna()
        st.success(f"🎉 최종 데이터셋 완성: {len(monthly_data.columns)}개 자산, {len(monthly_data)}개월 데이터")
        return monthly_data, substitution_log
    else:
        st.error("❌ 사용 가능한 데이터가 없습니다.")
        return None, {}

@st.cache_data
def load_universe_data_enhanced(tickers, start_date, end_date, fill_gaps=True):
    return fill_missing_data(tickers, start_date, end_date, fill_gaps)

@st.cache_data
def load_benchmark_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame(name=ticker)
        elif isinstance(data, pd.DataFrame) and len(data.columns) == 1:
            data.columns = [ticker]
        monthly_prices = data.resample('ME').last()
        monthly_prices = monthly_prices.dropna()
        if len(monthly_prices.columns) == 1:
            return monthly_prices.iloc[:, 0]
        else:
            return monthly_prices
    except Exception as e:
        st.error(f"벤치마크 데이터 로드 중 오류 발생: {str(e)}")
        return None

def adjust_weights_to_bounds(weights, upper_bound, lower_bound, max_iterations=100):
    adjusted_weights = weights.copy()
    for iteration in range(max_iterations):
        adjusted_weights = np.minimum(adjusted_weights, upper_bound)
        adjusted_weights = np.maximum(adjusted_weights, lower_bound)
        total_weight = adjusted_weights.sum()
        if abs(total_weight - 1.0) < 1e-6:
            break
        if total_weight > 1.0:
            excess = total_weight - 1.0
            adjustable_mask = adjusted_weights > lower_bound
            if adjustable_mask.sum() > 0:
                reduction = excess * (adjusted_weights / adjusted_weights[adjustable_mask].sum())
                reduction[~adjustable_mask] = 0
                adjusted_weights = adjusted_weights - reduction
                adjusted_weights = np.maximum(adjusted_weights, lower_bound)
        else:
            deficit = 1.0 - total_weight
            adjustable_mask = adjusted_weights < upper_bound
            if adjustable_mask.sum() > 0:
                addition = deficit * (adjusted_weights / adjusted_weights[adjustable_mask].sum())
                addition[~adjustable_mask] = 0
                adjusted_weights = adjusted_weights + addition
                adjusted_weights = np.minimum(adjusted_weights, upper_bound)
    adjusted_weights = adjusted_weights / adjusted_weights.sum()
    return adjusted_weights

def run_backtest(stock_returns, window, top_n_stocks, upper_bound, lower_bound):
    portfolio_returns = []
    portfolio_dates = []
    portfolio_composition = {}
    weights_composition = {}
    prev_weights = None
    progress_bar = st.progress(0)
    total_iterations = len(stock_returns) - window
    for i in range(window, len(stock_returns)):
        progress_bar.progress((i - window) / total_iterations)
        current_date = stock_returns.index[i]
        past_returns = stock_returns.iloc[i-window:i]
        momentum_score = (1 + past_returns).prod() - 1
        if i % 1 == 0:
            current_top_stocks = momentum_score.nlargest(top_n_stocks).index.tolist()
            portfolio_composition[current_date] = current_top_stocks
            lookback_period = min(36, i)
            if lookback_period < 12:
                continue
            historical_returns = stock_returns.iloc[i-lookback_period:i][current_top_stocks]
            if len(historical_returns) < 12:
                continue
            cov_matrix = historical_returns.cov()
            sigma_squared = np.diag(cov_matrix.values)
            sigma_squared = np.maximum(sigma_squared, 1e-8)
            momentum_scores = momentum_score[current_top_stocks].values
            momentum_scores = np.maximum(momentum_scores, 0.01)
            inverse_volatility = 1 / np.sqrt(sigma_squared)
            base_weights = inverse_volatility / inverse_volatility.sum()
            adjusted_weights = base_weights * np.sqrt(momentum_scores)
            adjusted_weights = adjusted_weights / adjusted_weights.sum()
            final_weights = adjust_weights_to_bounds(adjusted_weights, upper_bound, lower_bound)
            weights_composition[current_date] = dict(zip(current_top_stocks, final_weights))
            prev_weights = weights_composition[current_date]
        if prev_weights is not None:
            current_stocks = list(prev_weights.keys())
            weights_array = np.array(list(prev_weights.values()))
            available_returns = stock_returns.loc[current_date, current_stocks]
            if not available_returns.isna().any():
                portfolio_return = np.sum(weights_array * available_returns.values)
                portfolio_returns.append(portfolio_return)
                portfolio_dates.append(current_date)
            else:
                portfolio_returns.append(0.0)
                portfolio_dates.append(current_date)
    progress_bar.empty()
    return pd.Series(portfolio_returns, index=portfolio_dates), weights_composition

def safe_convert_to_float(value):
    try:
        if hasattr(value, 'item'):
            return float(value.item())
        elif hasattr(value, '__array__') and value.ndim == 0:
            return float(value)
        else:
            return float(value)
    except (ValueError, TypeError, AttributeError):
        return 0.0

def calculate_performance_metrics(returns, benchmark_returns=None):
    if len(returns) == 0:
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'tracking_error': 0.0
        }
    total_return = safe_convert_to_float((1 + returns).prod() - 1)
    annualized_return = safe_convert_to_float((1 + returns.mean())**12 - 1)
    volatility = safe_convert_to_float(returns.std() * np.sqrt(12))
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = safe_convert_to_float(drawdown.min())
    tracking_error = 0.0
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        common_index = returns.index.intersection(benchmark_returns.index)
        if len(common_index) > 1:
            aligned_returns = returns.loc[common_index]
            aligned_benchmark = benchmark_returns.loc[common_index]
            excess_returns = aligned_returns - aligned_benchmark
            tracking_error = safe_convert_to_float(excess_returns.std() * np.sqrt(12))
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'tracking_error': tracking_error
    }

def calculate_portfolio_turnover(weights_composition):
    if len(weights_composition) < 2:
        return 0.0
    dates = sorted(weights_composition.keys())
    turnovers = []
    for i in range(1, len(dates)):
        current_date = dates[i]
        previous_date = dates[i-1]
        current_weights = weights_composition[current_date]
        previous_weights = weights_composition[previous_date]
        all_assets = set(list(current_weights.keys()) + list(previous_weights.keys()))
        total_change = 0.0
        for asset in all_assets:
            current_weight = current_weights.get(asset, 0.0)
            previous_weight = previous_weights.get(asset, 0.0)
            total_change += abs(current_weight - previous_weight)
        turnover = total_change / 2.0
        turnovers.append(turnover)
    annual_turnover = np.mean(turnovers) * 12 if turnovers else 0.0
    return annual_turnover

def get_rebalancing_changes(current_weights, previous_weights):
    all_stocks = set(list(current_weights.keys()) + (list(previous_weights.keys()) if previous_weights else []))
    changes = {}
    for stock in all_stocks:
        current_weight = current_weights.get(stock, 0)
        previous_weight = previous_weights.get(stock, 0) if previous_weights else 0
        change = current_weight - previous_weight
        if abs(change) > 0.001:
            changes[stock] = {
                'previous': previous_weight,
                'current': current_weight,
                'change': change,
                'action': 'INCREASE' if change > 0 else 'DECREASE' if change < 0 else 'HOLD'
            }
    return changes

def create_performance_charts(portfolio_returns, benchmark_returns, benchmark_name):
    common_index = portfolio_returns.index.intersection(benchmark_returns.index)
    port_aligned = portfolio_returns.loc[common_index]
    bench_aligned = benchmark_returns.loc[common_index]
    yearly_port = port_aligned.groupby(port_aligned.index.year).apply(lambda x: (1 + x).prod() - 1)
    yearly_bench = bench_aligned.groupby(bench_aligned.index.year).apply(lambda x: (1 + x).prod() - 1)
    monthly_port = port_aligned.tail(24)
    monthly_bench = bench_aligned.tail(24)
    fig_yearly = go.Figure()
    years = yearly_port.index
    fig_yearly.add_trace(go.Bar(
        x=years,
        y=yearly_port * 100,
        name='포트폴리오',
        marker_color='deeppink',
        opacity=0.7
    ))
    fig_yearly.add_trace(go.Bar(
        x=years,
        y=yearly_bench * 100,
        name=benchmark_name,
        marker_color='royalblue',
        opacity=0.7
    ))
    fig_yearly.update_layout(
        title="연도별",
        xaxis_title="연도",
        yaxis_title="수익률 (%)",
        barmode='group',
        template="plotly_dark",
        height=400
    )
    fig_monthly = go.Figure()
    months = [f"{d.year}-{d.month:02d}" for d in monthly_port.index]
    fig_monthly.add_trace(go.Bar(
        x=months,
        y=monthly_port * 100,
        name='포트폴리오',
        marker_color='deeppink',
        opacity=0.7
    ))
    fig_monthly.add_trace(go.Bar(
        x=months,
        y=monthly_bench * 100,
        name=benchmark_name,
        marker_color='royalblue',
        opacity=0.7
    ))
    fig_monthly.update_layout(
        title=f"월별 (최근 {len(monthly_port)}개월)",
        xaxis_title="월",
        yaxis_title="수익률 (%)",
        barmode='group',
        template="plotly_dark",
        height=400,
        xaxis=dict(tickangle=45)
    )
    return fig_yearly, fig_monthly

def main():
    col_title, col_img_credit = st.columns([8, 1])
    with col_title:
        st.title("📈 Portfolio Backtesting App")
        #st.markdown("##### 만든이: 박석")
    with col_img_credit:
    # 닐 암스트롱 달착륙 사진(퍼블릭 도메인, NASA) - 다운로드 실패시 대체 아이콘 제공
        image_url = "https://cdn.theatlantic.com/thumbor/gjwD-uCiv0sHowRxQrQgL9b3Shk=/900x638/media/img/photo/2019/07/apollo-11-moon-landing-photos-50-ye/a01_40-5903/original.jpg"
        fallback_icon = "https://cdn-icons-png.flaticon.com/512/3211/3211357.png"  # 우주인 아이콘 (flaticon)
        img_displayed = False
        try:
            response = requests.get(image_url, timeout=5)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            st.image(img, width=200, caption=None)
            img_displayed = True
        except Exception:
            try:
                response = requests.get(fallback_icon, timeout=5)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
                st.image(img, width=200, caption=None)
                img_displayed = True
            except Exception:
                st.info("이미지를 불러올 수 없습니다.")


    #st.title("📈 Portfolio Backtesting App")
    #st.markdown("##### 만든이: 박석")
    


        st.markdown(
        "<div style='margin-top: -1px; text-align:center;'>"
        "<span style='font-size:0.9rem; color:#888;'>Made by parksuk1991</span>"
        "</div>",
        unsafe_allow_html=True
        )
        st.markdown(
            '<div style="text-align: right; margin-bottom: 9px;">'
            'Data 출처: <a href="https://finance.yahoo.com/" target="_blank">Yahoo Finance</a>'
            '</div>',
            unsafe_allow_html=True
        )

    
    with st.expander("📋 앱 소개", expanded=False):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("""
            **이 앱은 데이터 공백 자동 보완 기능을 갖춘 모멘텀 기반 포트폴리오 백테스팅 도구입니다.**
            #### 🎯 주요 기능
            - **자산 선택**: 원하는 ETF, 주식 등 자유로운 투자 유니버스 설정
            - **파라미터 조정**: 모멘텀 기간, 선택 종목 수, 최대/최소 가중치 등 전략 파라미터 조정
            - **기간 설정**: 백테스팅 분석 기간을 자유롭게 설정 가능
            - **모멘텀 전략**: 과거 수익률을 기준으로 상위 종목을 선별하여 포트폴리오를 구성
            - **리스크 조정**: 역변동성 가중치와 모멘텀 스코어를 결합한 스마트 포트폴리오 최적화
            - **월별 리밸런싱**: 매월 포트폴리오를 재조정하여 최적의 자산 구성 유지
            #### ✔️ 분석 결과 제공
            - **성과 지표**: 수익률, 변동성, 샤프 비율, 최대 낙폭 등 주요 투자 지표 분석
            - **벤치마크 비교**: S&P 500, Nasdaq 100, MSCI ACWI 지수와의 성과 비교
            - **시각화**: 누적 수익률, 리스크 분석, 포트폴리오 구성 변화 등 다양한 차트 제공
            #### 🔧 데이터 공백 보완 방식
            - **유사 종목 매핑**: 선택한 종목의 과거 데이터가 부족한 경우, 유사한 특성을 가진 대체 종목으로 자동 보완
            - **자산 분류별 대체**: 성장주, 가치주, 섹터별 등 자산 특성에 따른 체계적 대체
            """)
        with col2:
            st.markdown("""
            <div style="
                height: 600px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 15px;
                margin-top: 40px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                position: relative;
                overflow: hidden;
            ">
                <div style="
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: url('data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22%3E%3Cdefs%3E%3Cpattern id=%22grain%22 width=%22100%22 height=%22100%22 p[...]
                "></div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("---")
    st.sidebar.header("📊 유니버스 설정")
    default_tickers = [
        'XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLK', 'XLU',
        'SPYV', 'SPYG', 'VYM', 'RSP', 'USMV', 'SPMO', 'SPY', 'QQQ', 'IDEV', 'IEMG', 'ACWI', 'PTF', 'GRID', 'BOTZ', 'SMH', 'ITB', 
        'EWJ', 'IXUS', 'VGK', 'MCHI', 'EPP' 
    ]
    tickers_input = st.sidebar.text_area(
        "종목 티커 (쉼표로 구분)",
        value=", ".join(default_tickers[:35]),
        help="예시: SPY, QQQ, XLK",
        height=70
    )
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
    fill_gaps = st.sidebar.checkbox(
        "데이터 공백 보완 옵션",
        value=True,
        help="자산의 과거 데이터가 부족한 경우, 유사한 자산으로 자동 대체"
    )
    st.markdown("""
        <style>
        .stDateInput label { font-size: 13.5px !important; }
        .stDateInput input { font-size: 13.5px !important; }
        </style>
    """, unsafe_allow_html=True)
    st.sidebar.markdown('<div style="margin-top: 15px;"></div>', unsafe_allow_html=True)
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "시작 날짜",
            value=dt.date(2011, 1, 1),
            min_value=dt.date(2005, 1, 1),
            max_value=dt.date.today()
        )
    with col2:
        end_date = st.date_input(
            "종료 날짜",
            value=dt.date.today(),
            min_value=start_date,
            max_value=dt.date.today()
        )
    st.sidebar.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
    st.sidebar.subheader("백테스팅 설정")
    window = st.sidebar.slider("모멘텀 윈도우 (개월)", 3, 12, 6)
    top_n_stocks = st.sidebar.slider("선택 종목 수", 5, min(25, len(tickers)), min(15, len(tickers)))
    upper_bound = st.sidebar.slider("최대 가중치 (%)", 5, 20, 20) / 100
    lower_bound = st.sidebar.slider("최소 가중치 (%)", 1, 5, 1) / 100
    benchmark_display = st.sidebar.selectbox(
        "벤치마크",
        list(BENCHMARK_OPTIONS.keys()),
        index=2
    )
    benchmark_ticker = BENCHMARK_OPTIONS[benchmark_display]

    if st.sidebar.button("🚀 포트폴리오 생성", type="primary"):
        if len(tickers) < 5:
            st.error("최소 5개 이상의 티커를 입력해주세요.")
            return
        try:
            with st.spinner("데이터 로딩 및 전처리 중..."):
                monthly_df, substitution_log = load_universe_data_enhanced(
                    tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), fill_gaps
                )
                if monthly_df is None:
                    st.error("데이터 로드에 실패했습니다.")
                    return
                if substitution_log:
                    st.subheader("🔄 데이터 대체 로그")
                    substitute_df = pd.DataFrame([
                        {
                            '원본 자산': original,
                            '대체 자산': info['substitute'],
                            '대체 시작일': info['substitute_start'].strftime('%Y-%m-%d') if info['substitute_start'] else 'N/A',
                            '대체 방식': '유사자산'
                        }
                        for original, info in substitution_log.items()
                    ])
                    st.dataframe(substitute_df, use_container_width=True, hide_index=True)
                benchmark_data = load_benchmark_data(
                    benchmark_ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
                )
                if benchmark_data is None:
                    st.error("벤치마크 데이터 로드에 실패했습니다.")
                    return
                if isinstance(benchmark_data, pd.Series):
                    benchmark_df = benchmark_data
                elif isinstance(benchmark_data, pd.DataFrame):
                    benchmark_df = benchmark_data.iloc[:, 0]
                else:
                    st.error(f"예상치 못한 벤치마크 데이터 타입: {type(benchmark_data)}")
                    return
                common_start = max(monthly_df.index[0], benchmark_df.index[0])
                common_end = min(monthly_df.index[-1], benchmark_df.index[-1])
                monthly_df = monthly_df.loc[common_start:common_end]
                benchmark_df = benchmark_df.loc[common_start:common_end]
                stock_returns = monthly_df.pct_change().dropna()
                benchmark_returns = benchmark_df.pct_change().dropna()
                st.write(f"자산 데이터 길이: {len(stock_returns)}")
                st.write(f"벤치마크 데이터 길이: {len(benchmark_returns)}")
                st.write(f"벤치마크 데이터 타입: {type(benchmark_returns)}")
            with st.spinner("백테스팅 실행 중..."):
                portfolio_returns, weights_composition = run_backtest(
                    stock_returns, window, top_n_stocks, upper_bound, lower_bound
                )
                common_index = portfolio_returns.index.intersection(benchmark_returns.index)
                portfolio_returns_aligned = portfolio_returns.loc[common_index]
                benchmark_returns_aligned = benchmark_returns.loc[common_index]
                portfolio_metrics = calculate_performance_metrics(portfolio_returns_aligned, benchmark_returns_aligned)
                benchmark_metrics = calculate_performance_metrics(benchmark_returns_aligned)
                portfolio_turnover = calculate_portfolio_turnover(weights_composition)
            st.success(f"백테스팅 및 포트폴리오 생성 완료! ({common_index[0].strftime('%Y-%m')} ~ {common_index[-1].strftime('%Y-%m')})")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 포트폴리오 성과")
                benchmark_name = BENCHMARK_NAMES.get(benchmark_ticker, benchmark_ticker)
                metrics_df = pd.DataFrame({
                    '포트폴리오': [
                        f"{portfolio_metrics['total_return']:.2%}",
                        f"{portfolio_metrics['annualized_return']:.2%}",
                        f"{portfolio_metrics['volatility']:.2%}",
                        f"{portfolio_metrics['sharpe_ratio']:.2f}",
                        f"{portfolio_metrics['max_drawdown']:.2%}",
                        f"{portfolio_metrics['tracking_error']:.2%}"
                    ],
                    f'{benchmark_name}': [
                        f"{benchmark_metrics['total_return']:.2%}",
                        f"{benchmark_metrics['annualized_return']:.2%}",
                        f"{benchmark_metrics['volatility']:.2%}",
                        f"{benchmark_metrics['sharpe_ratio']:.2f}",
                        f"{benchmark_metrics['max_drawdown']:.2%}",
                        "N/A"
                    ]
                }, index=['총 수익률', '연평균 수익률', '연변동성', '샤프 비율', '최대 낙폭', '추적오차'])
                st.dataframe(metrics_df, use_container_width=True)
            with col2:
                st.subheader("📋 백테스팅 정보")
                info_df = pd.DataFrame({
                    '항목': ['분석 기간', '총 종목 수', '선택 종목 수', '리밸런싱', '가중치 범위', '연간 회전율'],
                    '값': [
                        f"{common_index[0].strftime('%Y-%m')} ~ {common_index[-1].strftime('%Y-%m')}",
                        f"{len(tickers)}개",
                        f"{top_n_stocks}개",
                        "매월",
                        f"{lower_bound:.1%} ~ {upper_bound:.1%}",
                        f"{portfolio_turnover:.1%}"
                    ]
                })
                st.dataframe(info_df, use_container_width=True, hide_index=True)
            st.subheader(f"📰 포트폴리오 업데이트 ({dt.date.today().strftime('%Y-%m')} 기준)")
            if weights_composition:
                recent_dates = sorted(weights_composition.keys())
                latest_date = recent_dates[-1]
                previous_date = recent_dates[-2] if len(recent_dates) > 1 else None
                current_weights = weights_composition[latest_date]
                previous_weights = weights_composition[previous_date] if previous_date else None
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**📕{latest_date.strftime('%Y-%m-%d')} 리밸런싱 안**")
                    current_df = pd.DataFrame([
                        {'종목': stock, '비중': f"{weight:.2%}"}
                        for stock, weight in sorted(current_weights.items(), key=lambda x: x[1], reverse=True)
                    ])
                    st.dataframe(current_df, use_container_width=True, hide_index=True)
                    fig_pie = px.pie(
                        values=list(current_weights.values()),
                        names=list(current_weights.keys()),
                        title="📒현재 비중 분포"
                    )
                    fig_pie.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
                with col2:
                    if previous_weights:
                        st.write(f"**📙전월 대비 리밸런싱 변화** ({previous_date.strftime('%Y-%m-%d')} → {latest_date.strftime('%Y-%m-%d')})")
                        changes = get_rebalancing_changes(current_weights, previous_weights)
                        if changes:
                            rebalancing_data = []
                            for stock, change_info in sorted(changes.items(), key=lambda x: abs(x[1]['change']), reverse=True):
                                action_emoji = "📈" if change_info['action'] == 'INCREASE' else "📉" if change_info['action'] == 'DECREASE' else "➡️"
                                rebalancing_data.append({
                                    '종목': f"{action_emoji} {stock}",
                                    '이전 비중': f"{change_info['previous']:.2%}",
                                    '현재 비중': f"{change_info['current']:.2%}",
                                    '변화': f"{change_info['change']:+.2%}"
                                })
                            rebalancing_df = pd.DataFrame(rebalancing_data)
                            st.dataframe(rebalancing_df, use_container_width=True, hide_index=True)
                            stocks = list(changes.keys())
                            changes_values = [changes[stock]['change'] for stock in stocks]
                            colors = ['deeppink' if x > 0 else 'royalblue' for x in changes_values]
                            fig_rebal = go.Figure(data=[
                                go.Bar(x=stocks, y=[x*100 for x in changes_values],
                                      marker_color=colors,
                                      text=[f"{x:+.1%}" for x in changes_values],
                                      textposition='auto')
                            ])
                            fig_rebal.update_layout(
                                title="📗리밸런싱 변화 (%p)",
                                xaxis_title="종목",
                                yaxis_title="비중 변화 (%p)",
                                template="plotly_dark",
                                height=400
                            )
                            st.plotly_chart(fig_rebal, use_container_width=True)
                        else:
                            st.info("이전 월 대비 유의미한 리밸런싱 변화가 없습니다.")
                    else:
                        st.info("비교할 이전 포트폴리오 데이터가 없습니다.")
            st.subheader("📈 성과 분석")
            benchmark_name = BENCHMARK_NAMES.get(benchmark_ticker, benchmark_ticker)
            st.write("=== 시각화 데이터 검증 ===")
            st.write(f"포트폴리오 수익률 샘플: {portfolio_returns_aligned.head(3).values}")
            st.write(f"벤치마크 수익률 샘플: {benchmark_returns_aligned.head(3).values}")
            st.write(f"포트폴리오 수익률 NaN 개수: {portfolio_returns_aligned.isna().sum()}")
            st.write(f"벤치마크 수익률 NaN 개수: {benchmark_returns_aligned.isna().sum()}")
            cumulative_portfolio = (1 + portfolio_returns_aligned).cumprod() - 1
            cumulative_benchmark = (1 + benchmark_returns_aligned).cumprod() - 1
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=cumulative_portfolio.index,
                y=cumulative_portfolio * 100,
                mode='lines',
                name='포트폴리오',
                line=dict(color='deeppink', width=2)
            ))
            fig1.add_trace(go.Scatter(
                x=cumulative_benchmark.index,
                y=cumulative_benchmark * 100,
                mode='lines',
                name=benchmark_name,
                line=dict(color='royalblue', width=2, dash='dash')
            ))
            fig1.update_layout(
                title="누적 수익률 비교",
                xaxis_title="날짜",
                yaxis_title="누적 수익률 (%)",
                hovermode='x unified',
                template="plotly_dark"
            )
            st.plotly_chart(fig1, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(
                    x=portfolio_returns_aligned * 100,
                    name='포트폴리오',
                    opacity=0.7,
                    marker_color='deeppink',
                    nbinsx=20
                ))
                fig2.add_trace(go.Histogram(
                    x=benchmark_returns_aligned * 100,
                    name=benchmark_name,
                    opacity=0.7,
                    marker_color='royalblue',
                    nbinsx=20
                ))
                fig2.update_layout(
                    title="월별 수익률 분포",
                    xaxis_title="월별 수익률 (%)",
                    yaxis_title="빈도",
                    barmode='overlay',
                    template="plotly_dark"
                )
                st.plotly_chart(fig2, use_container_width=True)
            with col2:
                rolling_sharpe_portfolio = portfolio_returns_aligned.rolling(12).mean() / portfolio_returns_aligned.rolling(12).std() * np.sqrt(12)
                rolling_sharpe_benchmark = benchmark_returns_aligned.rolling(12).mean() / benchmark_returns_aligned.rolling(12).std() * np.sqrt(12)
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=rolling_sharpe_portfolio.index,
                    y=rolling_sharpe_portfolio,
                    mode='lines',
                    name='포트폴리오',
                    line=dict(color='deeppink', width=2)
                ))
                fig3.add_trace(go.Scatter(
                    x=rolling_sharpe_benchmark.index,
                    y=rolling_sharpe_benchmark,
                    mode='lines',
                    name=benchmark_name,
                    line=dict(color='royalblue', width=2, dash='dash')
                ))
                fig3.update_layout(
                    title="12개월 롤링 샤프 비율",
                    xaxis_title="날짜",
                    yaxis_title="샤프 비율",
                    hovermode='x unified',
                    template="plotly_dark"
                )
                st.plotly_chart(fig3, use_container_width=True)
            portfolio_cumulative = (1 + portfolio_returns_aligned).cumprod()
            portfolio_running_max = portfolio_cumulative.expanding().max()
            portfolio_drawdown = (portfolio_cumulative - portfolio_running_max) / portfolio_running_max
            benchmark_cumulative = (1 + benchmark_returns_aligned).cumprod()
            benchmark_running_max = benchmark_cumulative.expanding().max()
            benchmark_drawdown = (benchmark_cumulative - benchmark_running_max) / benchmark_running_max
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=portfolio_drawdown.index,
                y=portfolio_drawdown * 100,
                fill='tonexty',
                mode='lines',
                name='포트폴리오',
                line=dict(color='deeppink', width=1),
                fillcolor='rgba(255,20,147,0.3)'
            ))
            fig4.add_trace(go.Scatter(
                x=benchmark_drawdown.index,
                y=benchmark_drawdown * 100,
                fill='tonexty',
                mode='lines',
                name=benchmark_name,
                line=dict(color='royalblue', width=1),
                fillcolor='rgba(65,105,225,0.3)'
            ))
            fig4.update_layout(
                title="낙폭 (Drawdown) 비교",
                xaxis_title="날짜",
                yaxis_title="낙폭 (%)",
                hovermode='x unified',
                template="plotly_dark"
            )
            st.plotly_chart(fig4, use_container_width=True)
            st.subheader("📅 연도별 및 월별 성과")
            fig_yearly, fig_monthly = create_performance_charts(
                portfolio_returns_aligned, benchmark_returns_aligned, benchmark_name
            )
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_yearly, use_container_width=True)
            with col2:
                st.plotly_chart(fig_monthly, use_container_width=True)
            st.subheader("📑 포트폴리오 구성 히스토리")
            if weights_composition:
                recent_dates = sorted(weights_composition.keys())[-5:]
                for date_key in recent_dates:
                    weights = weights_composition[date_key]
                    with st.expander(f"{date_key.strftime('%Y-%m-%d')} 포트폴리오 구성"):
                        weights_df = pd.DataFrame([
                            {'종목': stock, '가중치': f"{weight:.2%}"}
                            for stock, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True)
                        ])
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.dataframe(weights_df, use_container_width=True, hide_index=True)
                        with col2:
                            fig_pie = px.pie(
                                values=list(weights.values()),
                                names=list(weights.keys()),
                                title="가중치 분포"
                            )
                            fig_pie.update_layout(template="plotly_dark", height=300)
                            st.plotly_chart(fig_pie, use_container_width=True)
        except Exception as e:
            st.error(f"백테스팅 실행 중 오류가 발생했습니다: {str(e)}")
            import traceback
            st.error(f"상세 오류: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
