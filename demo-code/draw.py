import torch
import matplotlib.pyplot as plt

class DrawSample:
    @staticmethod
    def draw_samples(data, title="Sample Points"):
        plt.figure(figsize=(6,6))
        plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), color='blue', s=10)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
        plt.grid(False)
        plt.axis('equal')
        plt.show()
    
    @staticmethod
    def draw_samples_comparison(data1, data2, title1="Source Samples", title2="Target Samples"):
        plt.figure(figsize=(5,5))
        plt.scatter(data1[:, 0], data1[:, 1], color='blue', alpha=0.6, label=title1)
        plt.scatter(data2[:, 0], data2[:, 1], color='red', alpha=0.6, label=title2)
        plt.title("Source and Target samples")
        plt.legend()
        plt.show()
        
    
class DrawFlow:
    @staticmethod
    def draw_trajectory_from_results(results):
        trajectory = torch.stack(results, dim=0)  # (steps, batch, 2)

        plt.figure(figsize=(6,6))

        for i in range(results[0].shape[0]):
            # kknjknkj
            path = trajectory[:, i, :].cpu().detach().numpy()
            
            plt.plot(path[:, 0], path[:, 1], color='lightblue', linewidth=2, zorder=1)
            
            plt.scatter(path[0, 0], path[0, 1], color='blue', s=50, label='start' if i==0 else "", zorder=2)
            plt.scatter(path[-1, 0], path[-1, 1], color='red', s=50, label='end' if i==0 else "", zorder=2)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Quá trình dịch chuyển các điểm')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.show()
        
    @staticmethod
    def draw_gif():
        pass