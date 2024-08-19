import cv2
import numpy as np
import random

# محاسبه احتمال هر تصویر
def compute_probabilities(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    return hist

def fitness(image, threshold):
    # محاسبه پیش‌زمینه و پس‌زمینه
    foreground = image[image >= threshold]
    background = image[image < threshold]
    
    if len(foreground) == 0 or len(background) == 0:
        return 0
    
    # محاسبه میانگین سطوح خاکستری
    mean_foreground = np.mean(foreground)
    mean_background = np.mean(background)
    
    # محاسبه احتمالات
    probabilities = compute_probabilities(image)
    
    # محاسبه واریانس بین کلاسی
    between_class_variance = probabilities[int(mean_foreground)] * (mean_foreground - mean_background) ** 2
    
    return between_class_variance


def genetic_algorithm(image, population_size=20, generations=100, mutation_rate=0.01):
    height, width = image.shape
    
    # 1. تولید جمعیت اولیه
    population = []
    for _ in range(population_size):
        population.append(random.randint(50, 200))
    
    for generation in range(generations):
        # 2. محاسبه فتینس برای هر آستانه
        fitness_scores = []
        for threshold in population:
            score = fitness(image, threshold)
            fitness_scores.append((score, threshold))
        
        # مرتب‌سازی جمعیت بر اساس فیتنس
        fitness_scores.sort(reverse=True, key=get_fitness_score)
        
        # انتخاب بهترین آستانه‌ها برای نسل بعدی
        next_generation = [fitness_scores[0][1], fitness_scores[1][1]]
        
        while len(next_generation) < population_size:
            if random.random() < mutation_rate:
                # 3. جهش: تولید آستانه جدید به صورت تصادفی
                next_generation.append(random.randint(50, 200))
            else:
                # 4. کراس اور: ترکیب دو آستانه برای تولید آستانه جدید
                parents = random.sample(fitness_scores[:10], 2)
                child = int((parents[0][1] + parents[1][1]) / 2)
                next_generation.append(child)
        
        # 5. به روز رسانی جمعیت با نسل جدید
        population = next_generation
    
    # بهترین آستانه از نسل نهایی
    best_threshold = population[0]
    return best_threshold

# تابع کمکی برای مرتب‌سازی براساس فیتنس
def get_fitness_score(item):
    return item[0]

# خواندن تصویر و تبدیل به مقیاس خاکستری
image = cv2.imread('Picture1.png', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image,(500,500))

# اعمال الگوریتم ژنتیک برای پیدا کردن بهترین آستانه
best_threshold = genetic_algorithm(image)

# اعمال فیلتر سوبل و استفاده از آستانه بهینه برای تشخیص لبه‌ها
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel = np.hypot(sobel_x, sobel_y)
_, edges = cv2.threshold(sobel, best_threshold, 255, cv2.THRESH_BINARY)

# نمایش و ذخیره تصویر نهایی
cv2.imshow('Edges', edges)
cv2.imwrite('genetic_img3.jpg', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
