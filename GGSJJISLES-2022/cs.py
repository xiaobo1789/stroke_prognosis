import SimpleITK as sitk


fixed_path = "F:\stroke_prognosis\ISLES-2022\ISLES-2022\sub-strokecase0002\ses-0001\anat\sub-strokecase0002_ses-0001_FLAIR.nii.gz"  # 固定影像（如CT）
moving_path = "F:\stroke_prognosis\ISLES-2022\ISLES-2022\sub-strokecase0001\ses-0001\dwi\sub-strokecase0001_ses-0001_dwi.nii.gz"  # 浮动影像（如MRI）


fixed_img = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
moving_img = sitk.ReadImage(moving_path, sitk.sitkFloat32)


registration = sitk.ImageRegistrationMethod()
registration.SetMetricAsMeanSquares()  # 相似性度量（均方误差）
registration.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
registration.SetInitialTransform(sitk.TranslationTransform(fixed_img.GetDimension()))  # 初始变换（平移）


transform = registration.Execute(fixed_img, moving_img)

resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed_img)
resampler.SetTransform(transform)
registered_img = resampler.Execute(moving_img)

print("配准完成！输出影像尺寸与固定影像一致:", registered_img.GetSize() == fixed_img.GetSize())