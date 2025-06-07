# 预测启动脚本
from inference.predictor import Predictor

if __name__ == '__main__':
    predictor = Predictor.load('best_model.pth')
    result = predictor.predict('new_patient_data.csv')
    print(result)
