import optuna
import os
import plotly
import main_optuna

dir = r'/mnt/pycharm_project_VAE/optuna/20240221'

def objective(trial):
    # 这里应该是您的模型训练和评估逻辑
    return main_optuna.train_with_params(trial)


def visualize_study(study, trial, dir):
    trial_number = trial.number

    try:
        # 设定图像大小和分辨率
        width = 1600  # 图像宽度，可以根据需要进行调整
        height = 1200  # 图像高度，可以根据需要进行调整
        scale = 2  # 分辨率缩放因子，实际分辨率为 3200 × 2400 像素

        # 优化历史
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(os.path.join(dir, f'optimization_history_trial_{trial_number}.png'), width=width, height=height, scale=scale)

        # 参数重要性
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(os.path.join(dir, f'param_importances_trial_{trial_number}.png'), width=width, height=height, scale=scale)

        # 参数关联
        fig = optuna.visualization.plot_contour(study)
        fig.write_image(os.path.join(dir, f'contour_trial_{trial_number}.png'), width=width, height=height, scale=scale)

        # 切片图
        fig = optuna.visualization.plot_slice(study)
        fig.write_image(os.path.join(dir, f'slice_trial_{trial_number}.png'), width=width, height=height, scale=scale)

        print(f"试验 {trial_number} 的可视化结果已成功保存。")
    except Exception as e:
        print(f"试验 {trial_number} 的可视化结果保存失败：{e}")


def save_visualizations(study, trial):
    visualize_study(study, trial, dir)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, callbacks=[save_visualizations])
