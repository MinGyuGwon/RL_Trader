import random

import numpy as np
import utils


class Agent:
    # 에이전트 상태가 구성하는 값 개수
    STATE_DIM = 3  # 주식 보유 비율, 포트폴리오 가치 비율, 평균 매수 단가 대비 등락률

    # 매매 수수료 및 세금
    TRADING_CHARGE = 0.00015  # 거래 수수료 0.015%
    # TRADING_CHARGE = 0.00011  # 거래 수수료 0.011%
    # TRADING_CHARGE = 0  # 거래 수수료 미적용
    TRADING_TAX = 0.0025  # 거래세 0.25%
    # TRADING_TAX = 0  # 거래세 미적용

    # 행동
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 홀딩
    # 인공 신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_BUY, ACTION_SELL, ACTION_HOLD]
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수

    def __init__(
        self, environment, balance, min_trading_unit=1, max_trading_unit=2):
        # Environment 객체
        # 현재 주식 가격을 가져오기 위해 환경 참조
        self.environment = environment

        # 최소 매매 단위, 최대 매매 단위, 지연보상 임계치
        self.min_trading_unit = min_trading_unit  # 최소 단일 거래 단위
        self.max_trading_unit = max_trading_unit  # 최대 단일 거래 단위

        # Agent 클래스의 속성
        self.initial_balance = balance  # 초기 자본금
        self.balance = balance  # 현재 현금 잔고
        self.num_stocks = 0  # 보유 주식 수
        # 포트폴리오 가치: balance + num_stocks * {현재 주식 가격}
        self.portfolio_value = 0 
        self.base_portfolio_value = 0  # 직전 학습 시점의 PV
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 홀딩 횟수
        self.immediate_reward = 0  # 즉시 보상
        self.profitloss = 0  # 초기 자본금 대비 비율 ex_ +2% -2%
        self.base_profitloss = 0  # 직전 지연 보상 이후 손익 ???
        self.exploration_base = 0  # 탐험 행동 결정 기준

        # Agent 클래스의 상태
        self.ratio_hold = 0  # 주식 보유 비율
        self.ratio_portfolio_value = 0  # 포트폴리오 가치 비율
        self.avg_buy_price = 0  # 주당 매수 단가

    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    def reset_exploration(self, alpha=None):
        if alpha is None:
            alpha = 0
        self.exploration_base = 0.5 + alpha

    def set_balance(self, balance):
        self.initial_balance = balance

    def get_states(self):
        self.ratio_hold = self.num_stocks / int(self.portfolio_value / self.environment.get_price())
        self.ratio_portfolio_value = (  # 초기 자본금 대비 포트폴리오의 가치 비율  ex) 104%
            self.portfolio_value / self.base_portfolio_value  
        )
        return (
            self.ratio_hold, # 현재의 가격으로 소유할 수 있는 주식의 개수 중 소유한 주식 수의 비율
            self.ratio_portfolio_value, # 초기 자본금 대비 포트폴리오의 가치 비율
            (self.environment.get_price() / self.avg_buy_price) - 1 if self.avg_buy_price > 0 else 0 # 평균 매입가 대비 현재의 주식 가치의 '증감' 비율
        )

    def decide_action(self, pred_value, pred_policy, epsilon):
        confidence = 0.

        pred = pred_policy
        if pred is None:   # Policy Network이 작동을 안한다면 
            pred = pred_value  # Value Network에서 나온 값을 넣어라

        if pred is None:
            # 예측 값이 없을 경우 탐험
            epsilon = 1 # 탐험을 해라 
        else:
            # 값이 모두 같은 경우 탐험
            maxpred = np.max(pred)       # (pred == maxpred).all()  --> "모든 정책의 확률이 같다면" 
            if (pred == maxpred).all(): # 하나라도 거짓이면 False를 반환, 모두 True거나 iterable이 비어있으면 True를 반환
                epsilon = 1  # 모든 정책의 확률이 같은 경우에도 탐험을 해라~

        # 탐험 결정
        if np.random.rand() < epsilon: # 탐험을 하는 경우
            exploration = True
            if np.random.rand() < self.exploration_base:
                action = self.ACTION_BUY
            else:
                action = np.random.randint(self.NUM_ACTIONS - 1) + 1  # 관망이나 매도
        else: # 탐험을 하지 않는 경우
            exploration = False
            action = np.argmax(pred)  

        confidence = .5  # 매수, 매도의 '양'을 결정한다
        if pred_policy is not None:
            confidence = pred[action] # 정책 신경망 내 해당 액션의 확률
        elif pred_value is not None:
            confidence = utils.sigmoid(pred[action]) # 가치 신경망 내 해당 액션의 확률에 시그모이드 취해준다.

        return action, confidence, exploration

    def validate_action(self, action):
        if action == Agent.ACTION_BUY:
            # 적어도 1주를 살 수 있는지 확인
            if self.balance < self.environment.get_price() * (
                1 + self.TRADING_CHARGE) * self.min_trading_unit:
                return False
        elif action == Agent.ACTION_SELL:
            # 주식 잔고가 있는지 확인 
            if self.num_stocks <= 0:
                return False
        return True

    def decide_trading_unit(self, confidence):
        if np.isnan(confidence):
            return self.min_trading_unit
        added_traiding = max(min(   # min은 쓸 필요 없지 않나, 어차피 confidence가 1을 넘어갈리가 없는데?
            int(confidence * (self.max_trading_unit - 
                self.min_trading_unit)),
            self.max_trading_unit-self.min_trading_unit
        ), 0)
        return self.min_trading_unit + added_traiding

    def act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 즉시 보상 초기화
        self.immediate_reward = 0

        # 매수
        if action == Agent.ACTION_BUY:
            # 매수할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            balance = (
                self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            )
            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
            if balance < 0:
                trading_unit = max(
                    min(
                        int(self.balance / (
                            curr_price * (1 + self.TRADING_CHARGE))),
                        self.max_trading_unit
                    ),
                    self.min_trading_unit
                )
            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            if invest_amount > 0:
                self.avg_buy_price = (self.avg_buy_price * self.num_stocks + curr_price) / (self.num_stocks + trading_unit)  # 주당 매수 단가 갱신
                self.balance -= invest_amount  # 보유 현금을 갱신
                self.num_stocks += trading_unit  # 보유 주식 수를 갱신
                self.num_buy += 1  # 매수 횟수 증가

        # 매도
        elif action == Agent.ACTION_SELL:
            # 매도할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            trading_unit = min(trading_unit, self.num_stocks)
            # 매도
            invest_amount = curr_price * (
                1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
            if invest_amount > 0:
                self.avg_buy_price = (self.avg_buy_price * self.num_stocks - curr_price) / (self.num_stocks - trading_unit) if self.num_stocks > trading_unit else 0  # 주당 매수 단가 갱신
                self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
                self.balance += invest_amount  # 보유 현금을 갱신
                self.num_sell += 1  # 매도 횟수 증가

        # 홀딩
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 홀딩 횟수 증가

        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        self.profitloss = (
            (self.portfolio_value - self.initial_balance) / self.initial_balance
        )
        '''
        책에는 있던 내용
        # 즉시 보상 - 수익률
        self.immediate_reward = self.profitloss
        
        # 지연 보상 - 익절, 손절 기준
        delayed_reward = 0
        
        self.base_profitloss  = (
            (self.portfolio_value - self.base_portfolio_value) /
            self.base_portfolio_value
             )
        if self.base_profitloss > self.delayed_reward_threshold or \
            self.base_profitloss < -self.delayed_reward_threshold:
            # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
            # 또는 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신 
            self.base_portfolio_value = self.portfolio_value
            delayed_reward = self.immediate_reward
        else:
            delayed_reward = 0     
        '''
        return self.profitloss  # 책 내용상에서는 self.immediate_reward, delayed_reward를 반환
