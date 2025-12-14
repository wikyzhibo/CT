import pygame


nodes = {}
edges = []

# 初始化pygame
pygame.init()

# 定义颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# 设置窗口
WIDTH, HEIGHT = 1000, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Petri Net Animation')


# 初始token数
tokens = {
    'p1': 1, 'p2': 0, 'p3': 0, 'p4': 0, 'p5': 0,
    'p6': 0, 'p7': 0, 'p8': 0, 'p9': 0, 'p10': 0
}

fire_sequence = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8']





# 绘制Petri网图形
def draw_petri():
    screen.fill(WHITE)

    # 绘制变迁和库所
    for name in nodes:
        node = nodes[name]
        if node['type'] == 'p':
            x , y = node['x'], node['y']
            pygame.draw.circle(screen, BLACK, (x, y), 20, 2)  # 圆形表示库所
            font = pygame.font.SysFont(None, 24)
            text = font.render(name, True, BLACK)
            screen.blit(text, (x - 15, y - 10))
        elif node['type'] == 't':
            x, y = node['x'], node['y']
            pygame.draw.rect(screen, BLACK, (x - 20, y - 20, 40, 40), 2)  # 正方形表示变迁
            font = pygame.font.SysFont(None, 24)
            text = font.render(name, True, BLACK)
            screen.blit(text, (x - 15, y - 10))

    # 绘制弧
    '''
        for input,output,weight in edges:
        start_x, start_y = places[start] if input in nodes else transitions[start]
        end_x, end_y = places[end] if end in places else transitions[end]
        pygame.draw.line(screen, BLACK, (start_x, start_y), (end_x, end_y), 2)
        pygame.draw.polygon(screen, BLACK, [(end_x, end_y),
                                            (end_x - 5, end_y - 5),
                                            (end_x - 5, end_y + 5)])
                                            
    # 绘制token数
    font = pygame.font.SysFont(None, 24)
    for p, (x, y) in places.items():
        text = font.render(f'Token: {tokens[p]}', True, BLACK)
        screen.blit(text, (x - 25, y + 30))
    '''


# 更新Petri网状态
def update_petri(i):
    transition = fire_sequence[i]  # 获取当前发射的变迁
    if transition == 't1':
        tokens['p1'] -= 1
        tokens['p4'] += 1
        tokens['p5'] += 1
    elif transition == 't2':
        tokens['p4'] -= 1
        tokens['p6'] += 1
        tokens['p7'] += 1
    # 根据其他变迁逻辑继续更新...


# 游戏主循环
def main():


    clock = pygame.time.Clock()
    i = 0  # 变迁发射序列的索引
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 每隔一段时间更新一次变迁发射
        #if i < len(fire_sequence):
        #    update_petri(i)
        #    i += 1

        draw_petri()
        pygame.display.flip()
        clock.tick(1000)  # 控制更新的帧率为1，模拟动画效果

    pygame.quit()
    sys.exit()


# 启动程序
if __name__ == "__main__":
    main()
