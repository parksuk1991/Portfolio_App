# Portfolio_App

Streamlit 기반 Web Application prototype

Version.2025.06


1. 분석기간이 2019년 7월부터 시작되는 이유
1) 데이터 로딩 및 전처리
사용자가 시작 날짜를 2018/04/30으로 지정해도, 실제 포트폴리오 백테스트의 첫 결과가 나오는 시점은 더 뒤일 수 있습니다.
그 이유는 모멘텀 윈도우(window), lookback period, 그리고 최소 데이터 요구조건 때문입니다.
2) 핵심 코드 위치
run_backtest 함수에서 다음과 같이 동작합니다:
Apply to app_reposito...
설명
for i in range(window, len(stock_returns)):
→ 데이터의 첫 window(예: 6개월) 구간은 모멘텀 계산을 위해 사용되고, 실제 포트폴리오 수익률은 그 이후부터 계산됩니다.
lookback_period = min(36, i)
→ 최대 36개월(3년)까지 과거 데이터를 사용하지만, 데이터가 부족하면 현재 인덱스(i)만큼만 사용합니다.
if lookback_period < 12: continue
→ lookback period가 12개월 미만이면 포트폴리오를 구성하지 않고 건너뜁니다.
if len(historical_returns) < 12: continue
→ 실제로 과거 수익률 데이터가 12개월 미만이면 또 건너뜁니다.
즉,
데이터의 시작이 2018/04/30이라도,
window(예: 6개월)만큼은 모멘텀 계산용으로만 쓰이고,
lookback period가 12개월 이상이 되어야 실제 포트폴리오가 구성됩니다.
따라서 2018/04/30 + 6개월(window) + 12개월(lookback 최소값) = 2019/06/30 이후부터 포트폴리오가 생성될 수 있습니다.
실제로는 데이터 결측, 리샘플링 등으로 인해 2019년 7월부터 결과가 나오는 것입니다.
2. lookback period의 역할 및 포트폴리오 구성 로직
1) 모멘텀 기반 포트폴리오 구성
매월(혹은 주기별)로 과거 일정 기간(window) 동안의 수익률을 계산하여, 상위 N개 종목을 선정합니다.
선정된 종목들의 과거 12~36개월(lookback period) 수익률, 변동성 등을 바탕으로 가중치를 산정합니다.
2) 코드상 주요 흐름
(1) 모멘텀 스코어 계산
Apply to app_reposito...
최근 window(예: 6개월) 동안의 수익률을 누적 곱으로 계산하여 모멘텀 스코어 산출
상위 N개 종목 선정
(2) lookback period 적용
Apply to app_reposito...
lookback period는 최대 36개월, 최소 12개월
선정된 종목의 과거 lookback 기간 동안의 수익률 데이터가 12개월 미만이면 포트폴리오를 구성하지 않음
(3) 가중치 산정
변동성 역수 기반 가중치, 모멘텀 스코어 반영, 가중치 상하한 조정 등
