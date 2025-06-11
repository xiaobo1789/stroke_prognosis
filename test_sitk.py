import SimpleITK as sitk

# 生成浮点型测试影像（避免类型不兼容）
fixed = sitk.Image(128, 128, sitk.sitkFloat32)  # 固定影像（32位浮点型）
moving = sitk.Image(128, 128, sitk.sitkFloat32)  # 浮动影像（32位浮点型）

# 配置刚性配准（平移变换）
registration = sitk.ImageRegistrationMethod()
registration.SetMetricAsMeanSquares()  # 相似性度量（均方误差）
registration.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=10)
registration.SetInitialTransform(sitk.TranslationTransform(2))  # 2D 平移变换（维度2）

# 执行配准
try:
    transform = registration.Execute(fixed, moving)
    print("配准变换参数:", transform.GetParameters())
    print("SimpleITK 功能验证通过！")
except Exception as e:
    print(f"配准失败，错误信息: {e}")