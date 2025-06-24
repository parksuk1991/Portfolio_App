# -*- coding: utf-8 -*-
"""APP_repository.ipynb
"""


# Enhanced app.py with data gap filling functionality
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.stats import pearsonr
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

# 유사 자산 매핑
SIMILAR_ASSETS_MAP = {
    # 섹터 ETF 매핑
    'XLC': ['XTL', 'IYZ', 'VNQ'],  # Communication Services -> Telecom/Tech/REITs
    'XLY': ['RTH', 'XRT', 'VCR'],  # Consumer Discretionary -> Retail
    'XLP': ['VDC', 'PBJ', 'SZK'],  # Consumer Staples
    'XLE': ['VDE', 'IYE', 'DIG'],  # Energy
    'XLF': ['VFH', 'IYF', 'KBE'],  # Financials
    'XLV': ['VHT', 'IYH', 'PJP'],  # Healthcare
    'XLI': ['VIS', 'IYJ', 'PPA'],  # Industrials
    'XLB': ['VAW', 'IYM', 'SLX'],  # Materials
    'XLK': ['VGT', 'IYW', 'QQQ'],  # Technology
    'XLU': ['VPU', 'IDU', 'PUI'],  # Utilities

    # 스타일 ETF 매핑
    'SPYV': ['IVE', 'VTV', 'DVY'],  # S&P 500 Value
    'SPYG': ['IVW', 'VUG', 'MGK'],  # S&P 500 Growth
    'VYM': ['DVY', 'VTV', 'SCHD'],  # High Dividend Yield
    'RSP': ['EQL', 'EWRS', 'SPY'],  # Equal Weight S&P 500
    'USMV': ['SPLV', 'EFAV', 'SPY'],  # Low Volatility
    'SPMO': ['MTUM', 'PDP', 'QQQ'],  # Momentum

    # 리전 ETF 매핑
    'IDEV': ['EFA', 'VEA', 'ACWX'],  # Developed Markets
    'IEMG': ['EEM', 'VWO', 'SCHE'],  # Emerging Markets
}

# 대체 자산 풀
FALLBACK_ASSETS = {
    'large_cap_growth': ['QQQ', 'VUG', 'IVW'],
    'large_cap_value': ['VTV', 'IVE', 'DVY'],
    'small_cap': ['IWM', 'VB', 'IJR'],
    'international_dev': ['EFA', 'VEA', 'ACWX'],
    'international_em': ['EEM', 'VWO', 'DEM'],
    'sectors': ['XLK', 'XLF', 'XLV', 'XLE', 'XLI'],
    'broad_market': ['SPY', 'VTI', 'ITOT']
}

def get_asset_classification(ticker):
    """자산 분류 함수"""
    growth_etfs = ['SPYG', 'VUG', 'IVW', 'MGK', 'QQQ', 'XLK']
    value_etfs = ['SPYV', 'VTV', 'IVE', 'DVY', 'VYM']
    international_etfs = ['IDEV', 'EFA', 'VEA', 'IEMG', 'EEM', 'VWO']
    sector_etfs = ['XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLK', 'XLU']

    if ticker in growth_etfs:
        return 'large_cap_growth'
    elif ticker in value_etfs:
        return 'large_cap_value'
    elif ticker in international_etfs:
        return 'international_dev' if ticker in ['IDEV', 'EFA', 'VEA'] else 'international_em'
    elif ticker in sector_etfs:
        return 'sectors'
    else:
        return 'broad_market'

# 확장된 자산 풀
EXTENDED_ASSET_POOL = {
    'large_cap_us': ['SPY', 'VOO', 'IVV', 'VTI', 'ITOT', 'SPTM', 'SPLG'],
    'large_cap_growth': ['QQQ', 'VUG', 'IVW', 'MGK', 'SPYG', 'VONG', 'IWF'],
    'large_cap_value': ['VTV', 'IVE', 'DVY', 'SPYV', 'VONV', 'IWD', 'VYM'],
    'mid_cap': ['MDY', 'IJH', 'VO', 'IVOO', 'SPMD', 'IWR', 'VMOT'],
    'small_cap': ['IWM', 'VB', 'IJR', 'VTWO', 'SPSM', 'VBR', 'IWN'],
    'international_dev': ['EFA', 'IEUR', 'IXUS', 'VEA', 'IEFA', 'ACWX', 'IDEV', 'VTEB', 'SCHF'],
    'international_em': ['EEM', 'VWO', 'IEMG', 'SCHE', 'DEM', 'SPEM', 'EEMV'],
    'technology': ['XLK', 'QQQ', 'VGT', 'IYW', 'FTEC', 'SOXX', 'IGV'],
    'communications': ['XLC', 'XTL', 'IYZ'],
    'healthcare': ['XLV', 'VHT', 'IYH', 'FHLC', 'PJP', 'IHI', 'BBH'],
    'financials': ['XLF', 'VFH', 'IYF', 'FNCL', 'KBE', 'IAT', 'PFI'],
    'energy': ['XLE', 'VDE', 'IYE', 'FENY', 'DIG', 'IEO', 'PXE'],
    'materials': ['XLB', 'VAW', 'IYM', 'FMAT', 'SLX', 'IYZ', 'DBB'],
    'industrials': ['XLI', 'VIS', 'IYJ', 'FIDU', 'PPA', 'ITA', 'PRN'],
    'utilities': ['XLU', 'VPU', 'IDU', 'FUTY', 'PUI', 'JXI', 'RYU'],
    'consumer_disc': ['XLY', 'VCR', 'IYC', 'FDIS', 'RTH', 'XRT', 'PEJ'],
    'consumer_staples': ['XLP', 'VDC', 'IYK', 'FSTA', 'PBJ', 'SZK', 'KXI'],
    'real_estate': ['VNQ', 'IYR', 'SCHH', 'FREL', 'RWR', 'USRT', 'ICF'],
    'bonds': ['AGG', 'BND', 'IEFA', 'SCHZ', 'IEF', 'TLT', 'SHY'],
    'commodities': ['DJP', 'DBC', 'PDBC', 'GSG', 'COMT', 'BCI', 'RJA'],
    'minvol': ['USMV', 'SPLV', 'EFAV', 'IDLV'],
    'momentum': ['SPMO', 'MTUM', 'IMTM', 'PDP']
    
}

# 카테고리별 우선순위 설정 (같은 카테고리 내에서만 대체)
CATEGORY_PRIORITY = {
    'large_cap_us': ['VOO', 'IVV', 'VTI', 'SPY', 'ITOT', 'SPTM', 'SPLG'],
    'large_cap_growth': ['VUG', 'IVW', 'QQQ', 'MGK', 'SPYG', 'VONG', 'IWF'],
    'large_cap_value': ['VTV', 'IVE', 'DVY', 'SPYV', 'VONV', 'IWD', 'VYM'],
    'mid_cap': ['VO', 'IJH', 'MDY', 'IVOO', 'SPMD', 'IWR', 'VMOT'],
    'small_cap': ['VB', 'IJR', 'IWM', 'VTWO', 'SPSM', 'VBR', 'IWN'],
    'international_dev': ['EFA', 'IEUR', 'IXUS', 'VEA', 'IEFA', 'ACWX', 'IDEV', 'VTEB', 'SCHF'],
    'international_em': ['VWO', 'IEMG', 'EEM', 'SCHE', 'DEM', 'SPEM', 'EEMV'],
    'technology': ['VGT', 'XLK', 'IYW', 'QQQ', 'FTEC', 'SOXX', 'IGV'],
    'communications': ['XLC','XTL', 'IYZ'],
    'healthcare': ['VHT', 'XLV', 'IYH', 'FHLC', 'PJP', 'IHI', 'BBH'],
    'financials': ['VFH', 'XLF', 'IYF', 'FNCL', 'KBE', 'IAT', 'PFI'],
    'energy': ['VDE', 'XLE', 'IYE', 'FENY', 'DIG', 'IEO', 'PXE'],
    'materials': ['VAW', 'XLB', 'IYM', 'FMAT', 'SLX', 'IYZ', 'DBB'],
    'industrials': ['VIS', 'XLI', 'IYJ', 'FIDU', 'PPA', 'ITA', 'PRN'],
    'utilities': ['VPU', 'XLU', 'IDU', 'FUTY', 'PUI', 'JXI', 'RYU'],
    'consumer_disc': ['VCR', 'XLY', 'IYC', 'FDIS', 'RTH', 'XRT', 'PEJ'],
    'consumer_staples': ['VDC', 'XLP', 'IYK', 'FSTA', 'PBJ', 'SZK', 'KXI'],
    'real_estate': ['VNQ', 'IYR', 'SCHH', 'FREL', 'RWR', 'USRT', 'ICF'],
    'bonds': ['BND', 'AGG', 'SCHZ', 'IEF', 'TLT', 'SHY', 'IEFA'],
    'commodities': ['DBC', 'PDBC', 'DJP', 'GSG', 'COMT', 'BCI', 'RJA'],
    'minvol': ['USMV', 'SPLV', 'EFAV', 'IDLV'],
    'momentum': ['SPMO', 'MTUM', 'IMTM', 'PDP']
}

def get_enhanced_asset_classification(ticker):
    """향상된 자산 분류 - 더 세분화된 카테고리"""
    
    for category, tickers in EXTENDED_ASSET_POOL.items():
        if ticker in tickers:
            return category
    
    return 'large_cap_us'  # 기본값

def find_best_substitute_enhanced(target_ticker, available_data, start_date, end_date, min_correlation=0.3):
    """향상된 대체 자산 선택 - 동일 카테고리 내에서만 선택"""
    
    # 1단계: 타겟 티커의 카테고리 확인
    asset_category = get_enhanced_asset_classification(target_ticker)
    
    # 2단계: 동일 카테고리 내 후보 자산들 (우선순위 순서)
    category_candidates = CATEGORY_PRIORITY.get(asset_category, [])
    
    # 타겟 티커 제외
    candidates = [ticker for ticker in category_candidates if ticker != target_ticker]
    
    if not candidates:
        print(f"Warning: No substitute candidates found for {target_ticker} in category {asset_category}")
        return None, None
    
    # 3단계: 각 후보의 데이터 품질 및 적합성 평가
    best_candidates = []
    
    for candidate in candidates:
        try:
            # 후보 데이터 로드
            candidate_data = yf.download(candidate, start=start_date, end=end_date, progress=False)
            
            if candidate_data.empty:
                continue
                
            candidate_prices = candidate_data['Close'] if 'Close' in candidate_data.columns else candidate_data
            
            if len(candidate_prices) < 100:  # 최소 데이터 길이 요구사항 완화
                continue
            
            # 데이터 품질 검사
            data_completeness = candidate_prices.count() / len(candidate_prices)
            if data_completeness < 0.7:  # 70% 이상 데이터 완전성
                continue
            
            correlation_scores = []
            
            if len(available_data.columns) > 0:
                # 공통 기간 찾기
                common_period = candidate_prices.index.intersection(available_data.index)
                
                if len(common_period) > 50:  # 최소 겹치는 기간 완화
                    candidate_returns = candidate_prices.loc[common_period].pct_change().dropna()
                    
                    for existing_asset in available_data.columns:
                        existing_returns = available_data[existing_asset].loc[common_period].pct_change().dropna()
                        
                        # 공통 인덱스
                        common_idx = candidate_returns.index.intersection(existing_returns.index)
                        
                        if len(common_idx) > 30:  # 최소 공통 데이터 완화
                            try:
                                corr, p_value = pearsonr(
                                    candidate_returns.loc[common_idx].fillna(0),
                                    existing_returns.loc[common_idx].fillna(0)
                                )
                                
                                if not np.isnan(corr):
                                    correlation_scores.append(abs(corr))
                            except:
                                continue
            
            # 평균 상관관계 계산
            avg_correlation = np.mean(correlation_scores) if correlation_scores else 0
            
            # 데이터 길이 점수
            length_score = min(len(candidate_prices) / 1000, 1.0)  # 4년 기준 정규화
            
            # 우선순위 점수 (리스트에서 앞에 있을수록 높은 점수)
            priority_score = (len(candidates) - candidates.index(candidate)) / len(candidates)
            
            # 복합 점수 계산 (우선순위를 더 중요하게 반영)
            composite_score = (priority_score * 0.4) + (length_score * 0.3) + (data_completeness * 0.2) + (avg_correlation * 0.1)
            
            best_candidates.append({
                'ticker': candidate,
                'data': candidate_prices,
                'correlation': avg_correlation,
                'length_score': length_score,
                'completeness': data_completeness,
                'priority_score': priority_score,
                'composite_score': composite_score
            })
            
        except Exception as e:
            print(f"Error processing candidate {candidate}: {str(e)}")
            continue
    
    # 4단계: 최고 점수 대체 자산 선택
    if best_candidates:
        # 복합 점수 기준 정렬
        best_candidates.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # 가장 좋은 후보 선택
        best_candidate = best_candidates[0]
        
        print(f"Substituting {target_ticker} ({asset_category}) with {best_candidate['ticker']}")
        print(f"  - Data completeness: {best_candidate['completeness']:.2%}")
        print(f"  - Data length: {len(best_candidate['data'])} days")
        print(f"  - Average correlation: {best_candidate['correlation']:.3f}")
        
        return best_candidate['ticker'], best_candidate['data']
    
    # 5단계: 모든 후보가 실패한 경우 - 카테고리 내 첫 번째 대안 선택
    for candidate in candidates:
        try:
            fallback_data = yf.download(candidate, start=start_date, end=end_date, progress=False)
            if not fallback_data.empty:
                fallback_prices = fallback_data['Close'] if 'Close' in fallback_data.columns else fallback_data
                if len(fallback_prices) > 50:  # 최소 기준 완화
                    print(f"Using fallback substitute {candidate} for {target_ticker} (category: {asset_category})")
                    return candidate, fallback_prices
        except:
            continue
    
    return None, None


def fill_missing_data(tickers, start_date, end_date, fill_gaps=True):
    """데이터 공백 채우기"""

    st.info("📊 데이터 로딩 및 공백 분석 중...")

    # 원본 데이터 로드 시도
    original_data = {}
    missing_tickers = []
    data_info = {}

    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)['Close']

            if isinstance(data, pd.Series):
                data = data.to_frame(name=ticker)

            # 데이터 품질 확인
            data_start = data.first_valid_index()
            data_end = data.last_valid_index()
            data_length = len(data.dropna())

            # 목표 시작일과 실제 데이터 시작일 비교
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

    # 대체 자산 찾기 + 데이터 결합
    st.info("🔄 대체 자산 검색 및 데이터 결합 중...")

    substitution_log = {}
    enhanced_data = original_data.copy()

    # 기존 데이터 DataFrame으로 결합
    if len(enhanced_data) > 0:
        available_data = pd.concat(enhanced_data.values(), axis=1)
        available_data.columns = enhanced_data.keys()
    else:
        available_data = pd.DataFrame()


        substitute_ticker, substitute_data = find_best_substitute_enhanced(
            ticker, available_data, start_date, end_date
        )

        if substitute_ticker and substitute_data is not None:
            # 대체 데이터 처리
            if isinstance(substitute_data, pd.Series):
                substitute_data = substitute_data.to_frame(name=substitute_ticker)

            # 원본 티커 이름으로 컬럼명 변경
            substitute_df = substitute_data.copy()
            substitute_df.columns = [ticker]

            enhanced_data[ticker] = substitute_df
            substitution_log[ticker] = {
                'substitute': substitute_ticker,
                'original_start': data_info.get(ticker, {}).get('original_data', pd.DataFrame()).first_valid_index(),
                'substitute_start': substitute_data.first_valid_index(),
                'method': 'similar_asset'
            }

            # available_data 업데이트
            if len(available_data) == 0:
                available_data = substitute_df
            else:
                available_data = pd.concat([available_data, substitute_df], axis=1)
        else:
            st.error(f"❌ {ticker}: 적절한 대체 자산을 찾을 수 없습니다.")

    # 최종 데이터 결합
    if len(enhanced_data) > 0:
        final_data = pd.concat(enhanced_data.values(), axis=1)
        final_data.columns = enhanced_data.keys()

        # 월말 리샘플링
        monthly_data = final_data.resample('ME').last().dropna()

        st.success(f"🎉 최종 데이터셋 완성: {len(monthly_data.columns)}개 자산, {len(monthly_data)}개월 데이터")

        return monthly_data, substitution_log
    else:
        st.error("❌ 사용 가능한 데이터가 없습니다.")
        return None, {}

# 캐시된 데이터 로더
@st.cache_data
def load_universe_data_enhanced(tickers, start_date, end_date, fill_gaps=True):
    """유니버스 데이터"""
    return fill_missing_data(tickers, start_date, end_date, fill_gaps)
    
@st.cache_data
def load_benchmark_data(ticker, start_date, end_date):
    """벤치마크 데이터"""
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
    """가중치 조정 함수"""
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
    """백테스팅 실행"""
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

        if i % 1 == 0:  # 매월 리밸런싱
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
    """값을 안전하게 float로 변환"""
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
    """성과 지표 계산 (추적오차 포함)"""
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

    # 추적오차 계산
    tracking_error = 0.0
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        # 공통 인덱스 찾기
        common_index = returns.index.intersection(benchmark_returns.index)
        if len(common_index) > 1:
            aligned_returns = returns.loc[common_index]
            aligned_benchmark = benchmark_returns.loc[common_index]
            
            # 초과수익률의 표준편차 (연환산)
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
    """포트폴리오 회전율 계산"""
    if len(weights_composition) < 2:
        return 0.0
    
    dates = sorted(weights_composition.keys())
    turnovers = []
    
    for i in range(1, len(dates)):
        current_date = dates[i]
        previous_date = dates[i-1]
        
        current_weights = weights_composition[current_date]
        previous_weights = weights_composition[previous_date]
        
        # 모든 자산 리스트
        all_assets = set(list(current_weights.keys()) + list(previous_weights.keys()))
        
        # 가중치 변화량 계산
        total_change = 0.0
        for asset in all_assets:
            current_weight = current_weights.get(asset, 0.0)
            previous_weight = previous_weights.get(asset, 0.0)
            total_change += abs(current_weight - previous_weight)
        
        # 회전율은 변화량의 절반 (매도와 매수가 같은 양이므로)
        turnover = total_change / 2.0
        turnovers.append(turnover)
    
    # 연간 회전율로 환산 (월별 리밸런싱이므로 12를 곱함)
    annual_turnover = np.mean(turnovers) * 12 if turnovers else 0.0
    return annual_turnover



def get_rebalancing_changes(current_weights, previous_weights):
    """리밸런싱 변화 계산"""
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

# 4. 연도별/월별 성과 차트 생성 함수
def create_performance_charts(portfolio_returns, benchmark_returns, benchmark_name):
    """연도별 및 월별 성과 비교 차트 생성"""
    
    # 공통 기간 데이터
    common_index = portfolio_returns.index.intersection(benchmark_returns.index)
    port_aligned = portfolio_returns.loc[common_index]
    bench_aligned = benchmark_returns.loc[common_index]
    
    # 연도별 성과
    yearly_port = port_aligned.groupby(port_aligned.index.year).apply(lambda x: (1 + x).prod() - 1)
    yearly_bench = bench_aligned.groupby(bench_aligned.index.year).apply(lambda x: (1 + x).prod() - 1)
    
    # 월별 성과 (최근 24개월)
    monthly_port = port_aligned.tail(24)
    monthly_bench = bench_aligned.tail(24)
    
    # 연도별 성과 차트
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
    
    # 월별 성과 차트
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

# 메인 앱
def main():
    st.title("📈 Portfolio Backtesting App")
    st.markdown("##### 만든이: 박석")

    st.markdown(
        '<div style="text-align: right; margin-bottom: 10px;">'
        'Data 출처: <a href="https://finance.yahoo.com/" target="_blank">Yahoo Finance</a>'
        '</div>',
        unsafe_allow_html=True
    )

    # 앱 설명 섹션을 expander로 감싸기
    with st.expander("📋 앱 소개", expanded=False):
        # 앱 설명 섹션을 컬럼으로 분할
        col1, col2 = st.columns([3, 1])  # 3:1 비율로 분할
        
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
            - **상관관계 분석**: 기존 종목과 높은 상관관계를 가진 대체 종목 선택
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
                    background: url('data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22%3E%3Cdefs%3E%3Cpattern id=%22grain%22 width=%22100%22 height=%22100%22 patternUnits=%22userSpaceOnUse%22%3E%3Ccircle cx=%2225%22 cy=%2225%22 r=%221%22 fill=%22%23ffffff%22 opacity=%220.1%22/%3E%3Ccircle cx=%2275%22 cy=%2275%22 r=%221%22 fill=%22%23ffffff%22 opacity=%220.1%22/%3E%3Ccircle cx=%2275%22 cy=%2225%22 r=%220.5%22 fill=%22%23ffffff%22 opacity=%220.1%22/%3E%3Ccircle cx=%2225%22 cy=%2275%22 r=%220.5%22 fill=%22%23ffffff%22 opacity=%220.1%22/%3E%3C/pattern%3E%3C/defs%3E%3Crect width=%22100%22 height=%22100%22 fill=%22url(%23grain)%22/%3E%3C/svg%3E');
                "></div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 사이드바 설정
    st.sidebar.header("📊 유니버스 설정")
    
    # 기본 티커 목록
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

    # 데이터 공백 보완 옵션 추가
    fill_gaps = st.sidebar.checkbox(
        "데이터 공백 보완 옵션",
        value=True,
        help="자산의 과거 데이터가 부족한 경우, 유사한 자산으로 자동 대체"
    )

    # 날짜 설정 - 간격 조정

    st.markdown("""
        <style>
        /* date input 라벨 크기 줄이기 */
        .stDateInput label {
            font-size: 13.5px !important;
        }
        /* 달력 내부 텍스트 크기 줄이기 */
        .stDateInput input {
            font-size: 13.5px !important;
        }
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

    # 백테스팅 파라미터
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

    # 실행 버튼
    if st.sidebar.button("🚀 포트폴리오 생성", type="primary"):
        if len(tickers) < 5:
            st.error("최소 5개 이상의 티커를 입력해주세요.")
            return

        try:
            with st.spinner("데이터 로딩 및 전처리 중..."):
                #데이터 로더
                monthly_df, substitution_log = load_universe_data_enhanced(
                    tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), fill_gaps
                )

                if monthly_df is None:
                    st.error("데이터 로드에 실패했습니다.")
                    return

                # 대체 로그 표시
                if substitution_log:
                    st.subheader("🔄 데이터 대체 로그")
                    substitute_df = pd.DataFrame([
                        {
                            '원본 자산': original,
                            '대체 자산': info['substitute'],
                            '대체 시작일': info['substitute_start'].strftime('%Y-%m-%d') if info['substitute_start'] else 'N/A',
                            '대체 방식': '유사자산' if info['method'] == 'similar_asset' else '기타'
                        }
                        for original, info in substitution_log.items()
                    ])
                    st.dataframe(substitute_df, use_container_width=True, hide_index=True)

                # 벤치마크 데이터
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

                # 공통 기간 조정
                common_start = max(monthly_df.index[0], benchmark_df.index[0])
                common_end = min(monthly_df.index[-1], benchmark_df.index[-1])

                monthly_df = monthly_df.loc[common_start:common_end]
                benchmark_df = benchmark_df.loc[common_start:common_end]

                stock_returns = monthly_df.pct_change().dropna()
                benchmark_returns = benchmark_df.pct_change().dropna()

                # 디버깅 정보
                st.write(f"자산 데이터 길이: {len(stock_returns)}")
                st.write(f"벤치마크 데이터 길이: {len(benchmark_returns)}")
                st.write(f"벤치마크 데이터 타입: {type(benchmark_returns)}")

            with st.spinner("백테스팅 실행 중..."):
                # 백테스팅 실행
                portfolio_returns, weights_composition = run_backtest(
                    stock_returns, window, top_n_stocks, upper_bound, lower_bound
                )

                # 포트폴리오와 벤치마크의 공통 인덱스 찾기
                common_index = portfolio_returns.index.intersection(benchmark_returns.index)

                # 공통 인덱스로 재정렬
                portfolio_returns_aligned = portfolio_returns.loc[common_index]
                benchmark_returns_aligned = benchmark_returns.loc[common_index]

                # 디버깅 정보 출력
                #st.write(f"포트폴리오 수익률 데이터 포인트: {len(portfolio_returns_aligned)}")
                #st.write(f"벤치마크 수익률 데이터 포인트: {len(benchmark_returns_aligned)}")
                #st.write(f"공통 기간: {common_index[0].strftime('%Y-%m-%d')} ~ {common_index[-1].strftime('%Y-%m-%d')}")

                # 성과 지표 계산
                portfolio_metrics = calculate_performance_metrics(portfolio_returns_aligned, benchmark_returns_aligned)
                benchmark_metrics = calculate_performance_metrics(benchmark_returns_aligned)
                
                # 회전율 계산
                portfolio_turnover = calculate_portfolio_turnover(weights_composition)


            # 결과 표시
            st.success(f"백테스팅 및 포트폴리오 생성 완료! ({common_index[0].strftime('%Y-%m')} ~ {common_index[-1].strftime('%Y-%m')})")

            # 성과 지표 테이블
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 포트폴리오 성과")
                benchmark_name = BENCHMARK_NAMES.get(benchmark_ticker, benchmark_ticker)

                # 안전한 포맷팅 (이제 모든 값이 float이므로 안전함)
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

            # 포트폴리오 구성과 리밸런싱 정보
            st.subheader(f"📰 포트폴리오 업데이트 ({dt.date.today().strftime('%Y-%m')} 기준)")

            if weights_composition:
                # 최근 두 개 날짜 가져오기
                recent_dates = sorted(weights_composition.keys())
                latest_date = recent_dates[-1]
                previous_date = recent_dates[-2] if len(recent_dates) > 1 else None

                current_weights = weights_composition[latest_date]
                previous_weights = weights_composition[previous_date] if previous_date else None

                col1, col2 = st.columns(2)

                with col1:
                    # 현재 포트폴리오 구성
                    st.write(f"**📕{latest_date.strftime('%Y-%m-%d')} 리밸런싱 안**")

                    current_df = pd.DataFrame([
                        {'종목': stock, '비중': f"{weight:.2%}"}
                        for stock, weight in sorted(current_weights.items(), key=lambda x: x[1], reverse=True)
                    ])
                    st.dataframe(current_df, use_container_width=True, hide_index=True)

                    # 파이 차트
                    fig_pie = px.pie(
                        values=list(current_weights.values()),
                        names=list(current_weights.keys()),
                        title="📒현재 비중 분포"
                    )
                    fig_pie.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col2:
                    # 리밸런싱 변화
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

                            # 리밸런싱 변화 시각화
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

            # 차트 생성
            st.subheader("📈 성과 분석")
            benchmark_name = BENCHMARK_NAMES.get(benchmark_ticker, benchmark_ticker)

            # 데이터 유효성 검증
            st.write("=== 시각화 데이터 검증 ===")
            st.write(f"포트폴리오 수익률 샘플: {portfolio_returns_aligned.head(3).values}")
            st.write(f"벤치마크 수익률 샘플: {benchmark_returns_aligned.head(3).values}")
            st.write(f"포트폴리오 수익률 NaN 개수: {portfolio_returns_aligned.isna().sum()}")
            st.write(f"벤치마크 수익률 NaN 개수: {benchmark_returns_aligned.isna().sum()}")

            # 누적 수익률
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

            # 4개 차트 2x2로 배치
            col1, col2 = st.columns(2)

            with col1:
                # 월별 수익률 분포
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
                # 롤링 샤프 비율 (12개월)
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

            # 낙폭 비교 차트
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



            
            # 연도별 및 월별 성과 차트
            st.subheader("📅 연도별 및 월별 성과")

            fig_yearly, fig_monthly = create_performance_charts(
                portfolio_returns_aligned, benchmark_returns_aligned, benchmark_name
            )

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_yearly, use_container_width=True)
            with col2:
                st.plotly_chart(fig_monthly, use_container_width=True)




            # 포트폴리오 구성 히스토리
            st.subheader("📑 포트폴리오 구성 히스토리")

            if weights_composition:
                recent_dates = sorted(weights_composition.keys())[-5:]  # 최근 5개월

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
                            # 파이 차트
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
