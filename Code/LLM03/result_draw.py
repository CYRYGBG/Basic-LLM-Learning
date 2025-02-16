import pandas as pd
import matplotlib.pyplot as plt

# 1. 从Excel读取数据
file_path = r'D:\06.学习资料\llm course\Basic-LLM-Learning\Code\LLM03\Result.xlsx'  # 替换为你的文件路径
df = pd.read_excel(file_path)

# 2. 重命名列（确保列名匹配）
df = df.rename(columns={'微调方法': 'method', 'rank': 'rank', 'Final Flexible': 'Accu'})
data = df.to_dict('records')

# 3. 动态获取参数
ranks = sorted(df['rank'].unique())  # 自动提取唯一的rank值并按升序排列
methods = df['method'].unique().tolist()  # 确保顺序不变

# 4. 配置视觉参数
colors = ["#2a4f6e", "#3b6c9e", "#4d89cf", "#6aa3e0", "#8cb4eb"]  # 颜色列表
hatches = ['////', '||||', '----', '....', '++++']  # 纹理列表
bar_width = 0.15
group_spacing = 0.5

# 5. 创建画布
plt.figure(figsize=(12, 7), dpi=100)
ax = plt.gca()

# 6. 计算坐标逻辑
all_x, all_loss, bar_colors, bar_hatches = [], [], [], []
for group_idx, rank in enumerate(ranks):
    base_x = group_idx * (len(methods) * bar_width + group_spacing)
    for method_idx, method in enumerate(methods):
        x_pos = base_x + method_idx * bar_width
        all_x.append(x_pos)
        # 通过DataFrame直接查询提高效率（替代next遍历）
        loss_value = df[(df['rank']==rank) & (df['method']==method)]['Accu'].values[0]
        all_loss.append(loss_value)
        bar_colors.append(colors[method_idx % len(colors)])
        bar_hatches.append(hatches[method_idx % len(hatches)])

# 7. 绘制柱状图
bars = ax.bar(
    all_x, all_loss, 
    width=bar_width, 
    color=bar_colors, 
    edgecolor='white', 
    linewidth=0.5, 
    hatch=bar_hatches
)

# 8. 添加数值标签（优化位置计算）
for bar, loss in zip(bars, all_loss):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005, 
        f'{loss:.4f}', 
        ha='center', 
        va='bottom', 
        fontsize=8
    )

# 9. 美化布局
ax.set_xticks([group_idx * (len(methods)*bar_width + group_spacing) + (len(methods)*bar_width)/2 for group_idx in range(len(ranks))])
ax.set_xticklabels([str(rank) for rank in ranks])
ax.set_xlabel('Rank', fontweight='bold')
ax.set_ylabel('Accuracy(%)', fontweight='bold')
ax.set_title('Final Checkpoint Eval(Flexible)', fontsize=12, pad=20)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 10. 创建自适应图例
legend_elements = [
    plt.Rectangle((0,0), 1, 1, 
                  facecolor=colors[i], 
                  edgecolor='white', 
                  hatch=hatches[i], 
                  label=method) 
    for i, method in enumerate(methods)
]
ax.legend(handles=legend_elements, title='Methods', ncol=2, frameon=True, shadow=True)

plt.tight_layout()

plt.show()
# plt.savefig('Best Checkpoint Eval(Strcit).png')