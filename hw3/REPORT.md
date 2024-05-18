**Хранения промежуточного сжатого представления использует следующий формат файла:**
- заголовочные: width, height, compression_factor (все np.uint32)
- само сингулярное разложение по трём цветам (RGB) в трёх массивах, размера: `width compression_factor`, `compression_factor`,  `k * compression_factor`. Все числа в массивах типа float32.

**Изначальная картинка**

![start image](images/sample3.bmp)

**Запуск для разных методов и разных сжатий**

| Метод | N = 2 | N = 3 | N = 5 | N = 10 | N = 20 | N = 50 |
|-------|-------|-------|-------|--------|--------|--------|
| **numpy** | ![](images/sample_numpy_2.bmp) | ![](images/sample_numpy_3.bmp) | ![](images/sample_numpy_5.bmp) | ![](images/sample_numpy_10.bmp) | ![](images/sample_numpy_20.bmp) | ![](images/sample_numpy_50.bmp) |
| **simple** | ![](images/sample_simple_2.bmp) | ![](images/sample_simple_3.bmp) | ![](images/sample_simple_5.bmp) | ![](images/sample_simple_10.bmp) | ![](images/sample_simple_20.bmp) | ![](images/sample_simple_50.bmp) |
| **advanced** | ![](images/sample_advanced_2.bmp) | ![](images/sample_advanced_3.bmp) | ![](images/sample_advanced_5.bmp) | ![](images/sample_advanced_10.bmp) | ![](images/sample_advanced_20.bmp) | ![](images/sample_advanced_50.bmp) |

Метод **advanced** взят [отсюда](https://www.degruyter.com/document/doi/10.1515/jisys-2018-0034/html)

Такая же посмотрим на эту же картинку худшего качества и меньшего размера, сжата *numpy*:
![start image](images/sample1.bmp)

| N = 2 | N = 3 | N = 5 | N = 10 | N = 20 | N = 30 |
|-------|-------|-------|--------|--------|--------|
| ![](images/sample1_numpy_2.bmp) | ![](images/sample1_numpy_3.bmp) | ![](images/sample1_numpy_5.bmp) | ![](images/sample1_numpy_10.bmp) | ![](images/sample1_numpy_20.bmp) | ![](images/sample1_numpy_30.bmp) |

**Вывод**

Заметим что видимой глазу разницы в различных сжатиях нет. Картинка худжшего качества, как и ожидалось, при сжатии становится хужшего качества, чем боллее тяжелый аналог. *Numpy* работает заметно быстрее остальных.