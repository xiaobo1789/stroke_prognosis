# 预测启动脚本

from inference.predictor import StrokePredictor  # 正确导入类

if __name__ == '__main__':
    predictor = StrokePredictor.load('best_model.pth')  # 需在StrokePredictor中实现load方法
    result = predictor.predict(tabular_data, ct_post_scan)  # 传递治疗后CT
    print(result)