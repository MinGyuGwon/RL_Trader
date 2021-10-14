class Environment:
    PRICE_IDX = 4  # 종가의 위치  # 0: date,  1: open , 2: high , 3: low, 4: close,기ㅏ 5: volume, ....

    def __init__(self, chart_data=None):
        self.chart_data = chart_data
        self.observation = None # 현재 위치에서의 관측값 
        self.idx = -1  # 현재 위치 # -1로 초기화되어 있는 이유는?

    def reset(self):
        self.observation = None
        self.idx = -1

    def observe(self):
        '''
        data = pd.read_csv("~/aiffel/RLTrader/data/v2/005930.csv") 기준
        len(data) --> 976
        csv파일 확인 시 2~980까지 총 979개 
        '''
        if len(self.chart_data) > self.idx + 1:  # 왜 등호는 포함이 안되는거지? 당연 len()은 숫자를 1부터 센다.반면 index는 0부터 세기 때문에 
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation # 조건을 만족하는 경우
        return None # 조건을 만족하지 않는 경우

    def get_price(self):
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]
        return None

    def set_chart_data(self, chart_data):
        self.chart_data = chart_data
