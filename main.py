import os
import sys
import logging
import argparse
import json

import settings
import utils
import data_manager


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock_code', nargs='+', default=['051910']) ##< default 
    #{005380: "현대차", 005930: " 삼성전자", 015760:"한국전력", 035420:"네이버", 051910: "LG화학", 068270: "셀트리온"}
    parser.add_argument('--ver', choices=['v1', 'v2', 'v3'], default='v2') ##< default='v3'
    parser.add_argument('--rl_method', choices=['dqn', 'pg', 'ac', 'a2c', 'a3c', 'monkey'], default='dqn') ##< default='dqn'
    parser.add_argument('--net', choices=['dnn', 'lstm', 'cnn', 'monkey'], default='dnn')
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--discount_factor', type=float, default=0.95) # 수정
    # start_epsilon 은 별도로 run.(~, learning) learning 인자 값을 0으로 주지 않으면
    # 디폴트 설정인 dqn dnn 설정에서는 쓰이지 않음
    parser.add_argument('--start_epsilon', type=float, default=0)
    parser.add_argument('--balance', type=int, default=1000000000) # 10억 설정
    parser.add_argument('--num_epoches', type=int, default=1000) #  10만 Epoch
    parser.add_argument('--backend', choices=['tensorflow', 'plaidml'], default='tensorflow')
    parser.add_argument('--output_name', default=utils.get_time_str())  # 한국 시간대비 2시간 빠르다
    parser.add_argument('--value_network_name') # 'parser.add_argument('--value_network_name')'은 None
    parser.add_argument('--policy_network_name')
    parser.add_argument('--reuse_models', action='store_true')
    parser.add_argument('--learning', action='store_false') #
    parser.add_argument('--start_date', default='20170101')
    parser.add_argument('--end_date', default='20171231')
    args = parser.parse_args()
    
    
    # Keras Backend 설정
    if args.backend == 'tensorflow':
        os.environ['KERAS_BACKEND'] = 'tensorflow'
    elif args.backend == 'plaidml':
        os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

    # 출력 경로 설정
    output_path = os.path.join(settings.BASE_DIR, 
        'output/{}_{}_{}'.format(args.output_name, args.rl_method, args.net))
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # 파라미터 기록
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(json.dumps(vars(args)))
    
    # 로그 기록 설정
    file_handler = logging.FileHandler(filename=os.path.join(
        output_path, "{}.log".format(args.output_name)), encoding='utf-8')
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s",
        handlers=[file_handler, stream_handler], level=logging.DEBUG)
        
    # 로그, Keras Backend 설정을 먼저하고 RLTrader 모듈들을 이후에 임포트해야 함
    from agent import Agent
    from learners import ReinforcementLearner, DQNLearner, \
        PolicyGradientLearner, ActorCriticLearner, A2CLearner, A3CLearner

  
    # 모델 경로 준비
    value_network_path = ''
    policy_network_path = ''
    if args.value_network_name is not None:
        value_network_path = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(args.value_network_name))
    else:
        # args.value_network_name이 None인 경우 '20211014053010_dqn_dnn_value.h5'과 같이 저장 
        value_network_path = os.path.join(output_path, '{}_{}_{}_value.h5'.format(args.output_name, args.rl_method, args.net)) 
    if args.policy_network_name is not None:
        policy_network_path = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(args.policy_network_name))
    else:
        policy_network_path = os.path.join(output_path, '{}_{}_{}_policy.h5'.format(args.output_name, args.rl_method, args.net))

    common_params = {}
    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_unit = []
    list_max_trading_unit = []

    for stock_code in args.stock_code:
        # 차트 데이터, 학습 데이터 준비
        print(stock_code)
        chart_data, training_data = data_manager.load_data(
            stock_code, args.start_date, args.end_date, ver=args.ver) # args.ver은 2가 기본
        
        # 최소/최대 투자 단위 설정
        min_trading_unit = max(int(100000 / chart_data.iloc[-1,:]['close']), 1) # 마지막 '행'의 모든 '열'의 데이터 중에서 ['close'] 열의 값
        max_trading_unit = max(int(1000000 / chart_data.iloc[-1,:]['close']), 1)
        
        # 공통 파라미터 설정
        common_params = {'rl_method': args.rl_method, 
            'net': args.net, 'num_steps': args.num_steps, 'lr': args.lr,
            'balance': args.balance, 'num_epoches': args.num_epoches, 
            'discount_factor': args.discount_factor, 'start_epsilon': args.start_epsilon,
            'output_path': output_path, 'reuse_models': args.reuse_models}

        # 강화학습 시작
        learner = None
        # args.rl_method가 'a3c'가 아니라면 실행되는 코드 블록
        if args.rl_method != 'a3c':
            common_params.update({'stock_code': stock_code,
                'chart_data': chart_data, 
                'training_data': training_data,
                'min_trading_unit': min_trading_unit, 
                'max_trading_unit': max_trading_unit})
            if args.rl_method == 'dqn':
                learner = DQNLearner(**{**common_params, 
                'value_network_path': value_network_path})  
            elif args.rl_method == 'pg':
                learner = PolicyGradientLearner(**{**common_params, 
                'policy_network_path': policy_network_path})
            elif args.rl_method == 'ac':
                learner = ActorCriticLearner(**{**common_params, 
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'a2c':
                learner = A2CLearner(**{**common_params, 
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'monkey':
                args.net = args.rl_method
                args.num_epoches = 1
                args.discount_factor = None
                args.start_epsilon = 1
                args.learning = False
                learner = ReinforcementLearner(**common_params)
            # 앞선 if, elif문을 통해 생성된 learner 객체를 통해 학습 진행 
            if learner is not None: 
                learner.run(learning=args.learning)
                learner.save_models()
        else:
            list_stock_code.append(stock_code)
            list_chart_data.append(chart_data)
            list_training_data.append(training_data)
            list_min_trading_unit.append(min_trading_unit)
            list_max_trading_unit.append(max_trading_unit)

    if args.rl_method == 'a3c':
        learner = A3CLearner(**{
            **common_params, 
            'list_stock_code': list_stock_code, 
            'list_chart_data': list_chart_data, 
            'list_training_data': list_training_data,
            'list_min_trading_unit': list_min_trading_unit, 
            'list_max_trading_unit': list_max_trading_unit,
            'value_network_path': value_network_path, 
            'policy_network_path': policy_network_path})
        learner.run(learning=args.learning)
        learner.save_models()