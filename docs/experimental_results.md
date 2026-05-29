# Kết quả thực nghiệm mô hình DQN cho bài toán giao dịch cổ phiếu Việt Nam

## Tóm tắt

Tài liệu này tổng hợp và phân tích kết quả thực nghiệm từ quá trình đánh giá mô hình Deep Q-Network (DQN), trong đó các kết quả huấn luyện được hợp nhất để so sánh các cấu hình siêu tham số. Tập thực nghiệm bao gồm 34 lần chạy hợp lệ trên bốn mã cổ phiếu Việt Nam: VNM, FPT, VIC và HPG, với dữ liệu lấy từ nguồn VCI.

Kết quả cho thấy cấu hình `tau_01_run1` đạt hiệu năng tốt nhất theo Sharpe ratio trên tập kiểm tra, với lợi nhuận trung bình +6.03%, Sharpe trung bình +0.6008, win rate 59.7%, maximum drawdown -5.40% và profit factor 2.17. Tuy nhiên, hiệu năng của mô hình không đồng đều giữa các mã cổ phiếu: VIC đóng góp phần lớn vào kết quả tích cực, trong khi VNM và FPT cho lợi nhuận âm. Điều này cho thấy mô hình có tiềm năng khai thác tín hiệu giao dịch, nhưng cần thêm kiểm định về tính ổn định, độ bền vững và khả năng tổng quát hóa trước khi có thể kết luận về khả năng ứng dụng thực tế.

## 1. Thiết kế thực nghiệm

### 1.1 Mục tiêu

Mục tiêu của thực nghiệm là đánh giá ảnh hưởng của các siêu tham số trong mô hình DQN đối với hiệu quả giao dịch cổ phiếu. Việc đánh giá tập trung vào khả năng sinh lời, lợi nhuận điều chỉnh theo rủi ro, tỷ lệ giao dịch thắng, mức sụt giảm vốn và độ ổn định của quá trình huấn luyện.

Quy trình đánh giá thực hiện bốn nhiệm vụ chính:

1. Thu thập các lần chạy huấn luyện hợp lệ từ các thư mục `artifacts/outputs/output_*` có `logs.json`.
2. Trích xuất chữ ký cấu hình từ các siêu tham số quan trọng trong `parameters`.
3. Tạo tên cấu hình rút gọn từ toàn bộ giá trị tham số thực tế, thay vì chỉ mô tả khác biệt so với baseline.
4. Khử trùng lặp các cấu hình có cùng chữ ký, tổng hợp bảng kết quả và trực quan hóa bằng heatmap, biểu đồ cột nhóm theo tham số và đường cong huấn luyện top cấu hình.

### 1.2 Dữ liệu

Thực nghiệm sử dụng bốn mã cổ phiếu:

| Mã cổ phiếu | Vai trò trong đánh giá |
|---|---|
| VNM | Đại diện nhóm cổ phiếu vốn hóa lớn ngành hàng tiêu dùng |
| FPT | Đại diện nhóm công nghệ và tăng trưởng |
| VIC | Đại diện nhóm bất động sản/vốn hóa lớn |
| HPG | Đại diện nhóm công nghiệp và vật liệu cơ bản |

Nguồn dữ liệu là VCI. Dữ liệu được chia theo tỷ lệ 80% cho huấn luyện, 10% cho validation và phần còn lại cho test.

### 1.3 Môi trường giao dịch

Môi trường giao dịch được cấu hình để phản ánh một số ràng buộc thực tế của thị trường chứng khoán Việt Nam, bao gồm chi phí giao dịch, thuế bán, trượt giá, lô giao dịch và chu kỳ T+2.

| Thành phần | Giá trị |
|---|---:|
| Vốn ban đầu | 1,000,000,000 |
| Window quan sát | 20 |
| Phí giao dịch | 0.0015 |
| Thuế bán | 0.001 |
| Slippage | 0.001 |
| ATR stop-loss multiplier | 1.5 |
| ATR take-profit multiplier | 3.0 |
| Rủi ro mỗi giao dịch | 0.02 |
| Stop loss | 0.07 |
| Take profit | 0.15 |
| Số ngày nắm giữ tối đa | 30 |
| Chu kỳ thanh toán | T+2 |
| Đơn vị lô | 100 |
| Biên độ giá | 0.07 |

### 1.4 Ý nghĩa các tham số môi trường giao dịch

Trong bài toán giao dịch cổ phiếu, cấu hình môi trường quy định trực tiếp cách tác nhân nhìn thấy thị trường, cách lệnh được thực thi và cách rủi ro được giới hạn. Các tham số này giúp thực nghiệm gần hơn với điều kiện giao dịch thực tế trên thị trường Việt Nam, thay vì đánh giá mô hình trong một môi trường lý tưởng không có chi phí.

| Tham số | Ý nghĩa | Vai trò trong thực nghiệm |
|---|---|---|
| `window` | Số phiên lịch sử được dùng làm cửa sổ quan sát | Quyết định độ dài ngữ cảnh thị trường mà agent nhận được tại mỗi thời điểm |
| `initial_cap` | Vốn khởi tạo của danh mục | Tạo mốc quy đổi lợi nhuận, drawdown và kích thước vị thế |
| `tx_cost` | Phí giao dịch khi mua/bán | Làm kết quả thực tế hơn và tránh chiến lược giao dịch quá nhiều lần |
| `sell_tax` | Thuế khi bán cổ phiếu | Mô phỏng ràng buộc chi phí riêng của lệnh bán trên thị trường Việt Nam |
| `slippage` | Độ lệch giá khớp lệnh so với giá quan sát | Phản ánh rủi ro khớp lệnh không đúng giá lý thuyết |
| `atr_sl_mult` | Hệ số cắt lỗ động dựa trên ATR | Điều chỉnh ngưỡng cắt lỗ theo biến động của từng mã cổ phiếu |
| `atr_tp_mult` | Hệ số chốt lời động dựa trên ATR | Điều chỉnh mức chốt lời theo biến động, tạo tỷ lệ reward/risk động |
| `risk_per_trade` | Tỷ lệ vốn tối đa có thể rủi ro trên mỗi giao dịch | Giới hạn kích thước vị thế để tránh tập trung rủi ro quá lớn |
| `stop_loss` | Mức cắt lỗ cố định tối đa | Đóng vai trò fallback khi ngưỡng ATR không phù hợp hoặc quá rộng |
| `take_profit` | Mức chốt lời cố định tối đa | Giới hạn mức kỳ vọng lợi nhuận trên mỗi vị thế |
| `max_hold` | Số phiên nắm giữ tối đa | Buộc agent đóng vị thế nếu tín hiệu không còn hiệu quả sau một khoảng thời gian |
| `t_plus` | Chu kỳ thanh toán T+ | Mô phỏng ràng buộc thanh toán, tránh hành vi mua bán phi thực tế |
| `lot_size` | Đơn vị giao dịch tối thiểu | Đảm bảo số lượng cổ phiếu tuân thủ lô giao dịch |
| `price_limit` | Biên độ tăng/giảm giá tối đa trong một phiên | Mô phỏng giới hạn trần/sàn của sàn HOSE |

Những tham số này ảnh hưởng đến cả lợi nhuận và rủi ro. Ví dụ, tăng chi phí giao dịch hoặc slippage thường làm giảm Return %, trong khi thay đổi `atr_sl_mult`, `atr_tp_mult` và `risk_per_trade` có thể làm thay đổi trực tiếp maximum drawdown, win rate và profit factor.

## 2. Mô hình và cấu hình huấn luyện

### 2.1 Kiến trúc tác nhân

Tác nhân được sử dụng là DQN với mạng neural gồm ba lớp ẩn. Cấu hình của run tốt nhất được ghi nhận như sau:

| Thành phần | Giá trị |
|---|---:|
| Loại agent | DQN |
| Hidden layers | [256, 128, 64] |
| Learning rate | 0.0005 |
| Learning rate decay | 0.995 |
| Learning rate tối thiểu | 0.00005 |
| Discount factor gamma | 0.97 |
| Target update tau | 0.01 |
| Epsilon start | 1.0 |
| Epsilon end | 0.03 |
| Epsilon decay | 0.997 |
| Replay buffer capacity | 20,000 |
| Batch size | 256 |
| Warmup steps | 1,000 |
| Weight decay | 0.0001 |
| Gradient clipping | 1.0 |
| Loss function | Huber |

Run tốt nhất sử dụng mô hình ngôn ngữ `mistral-small-4-119b-2603` và mô hình embedding `perplexity-ai/pplx-embed-context-v1-4b`, với kích thước embedding 2560.

### 2.2 Ý nghĩa các tham số của DQN agent

Bảng sau trình bày vai trò của các tham số DQN trong cấu hình agent và các tham số được thay đổi trong quá trình khảo sát siêu tham số.

| Tham số | Ý nghĩa | Tác động kỳ vọng |
|---|---|---|
| `type` | Loại tác nhân học tăng cường | Trong thực nghiệm này sử dụng DQN, phù hợp với không gian hành động rời rạc như mua/bán/giữ |
| `hidden` | Kích thước các lớp ẩn của Q-network | Mạng lớn hơn có khả năng biểu diễn phức tạp hơn, nhưng tăng rủi ro overfitting và chi phí tính toán |
| `lr` | Learning rate của optimizer | Giá trị cao giúp học nhanh nhưng dễ dao động; giá trị thấp ổn định hơn nhưng hội tụ chậm |
| `lr_decay` | Hệ số giảm learning rate theo episode | Giảm bước cập nhật khi gần hội tụ, giúp ổn định quá trình học |
| `lr_min` | Sàn learning rate | Tránh learning rate giảm quá nhỏ làm mô hình gần như ngừng học |
| `gamma` | Discount factor cho phần thưởng tương lai | `gamma` cao ưu tiên lợi ích dài hạn; `gamma` thấp ưu tiên tín hiệu ngắn hạn |
| `tau` | Tốc độ soft update target network | `tau` lớn làm target network cập nhật nhanh hơn; `tau` nhỏ giúp mục tiêu Q ổn định hơn |
| `eps_start` | Epsilon ban đầu trong epsilon-greedy | Xác định mức độ exploration đầu quá trình huấn luyện |
| `eps_end` | Epsilon tối thiểu | Đảm bảo agent vẫn giữ một mức exploration nhỏ khi đã hội tụ |
| `eps_decay` | Tốc độ giảm epsilon | Giá trị gần 1 làm exploration kéo dài hơn; giá trị nhỏ làm agent khai thác chính sách học được sớm hơn |
| `buffer_cap` | Dung lượng replay buffer | Buffer lớn tăng đa dạng kinh nghiệm, nhưng có thể lưu nhiều mẫu cũ không còn phù hợp |
| `batch_size` | Số mẫu trong mỗi lần cập nhật gradient | Batch lớn giảm nhiễu gradient nhưng tăng chi phí tính toán; batch nhỏ linh hoạt hơn nhưng nhiễu hơn |
| `warmup` | Số step trước khi bắt đầu học | Bảo đảm replay buffer có đủ mẫu để cập nhật ổn định |
| `weight_decay` | L2 regularization | Giảm overfitting của Q-network |
| `grad_clip` | Giới hạn norm gradient | Giảm nguy cơ exploding gradient trong quá trình học |
| `loss_fn` | Hàm mất mát khi ước lượng Q-value | Huber loss bền vững hơn MSE khi target Q có outlier |

### 2.3 Quy trình huấn luyện

| Thành phần | Giá trị |
|---|---:|
| Số episode tối đa | 500 |
| Patience early stopping | 80 |
| Tần suất checkpoint | 50 episodes |
| Tần suất validation | 5 episodes |
| Tần suất học | 4 steps |
| Preload embeddings | true |

Run tốt nhất `tau_01_run1` đạt best score tại episode 75, dừng sớm sau 401 episodes và có tổng thời gian chạy 77.78 phút.

### 2.4 Ý nghĩa các tham số huấn luyện

| Tham số | Ý nghĩa | Vai trò trong quy trình đánh giá |
|---|---|---|
| `n_episodes` | Số episode huấn luyện tối đa | Quy định ngân sách học lớn nhất cho mỗi run |
| `patience` | Số lần validation không cải thiện trước khi dừng sớm | Giảm rủi ro overfitting và tiết kiệm chi phí tính toán |
| `checkpoint_every` | Tần suất lưu checkpoint | Cho phép truy xuất và khôi phục mô hình trong quá trình huấn luyện |
| `val_every` | Tần suất đánh giá trên validation | Quyết định độ mịn của tín hiệu early stopping và chọn best model |
| `learn_every` | Số step mới thực hiện một lần cập nhật gradient | Điều khiển tần suất học; giá trị nhỏ cập nhật thường xuyên hơn |
| `preload_embeddings` | Nạp trước embedding vào RAM | Giảm thời gian I/O khi huấn luyện với đặc trưng embedding |
| `show_step_progress` | Hiển thị tiến trình chi tiết bên trong từng step | Hữu ích khi theo dõi quá trình huấn luyện, nhưng có thể làm đầu ra thực nghiệm dài hơn |

## 3. Không gian siêu tham số

Các thực nghiệm trong `main.ipynb` được thiết kế theo nguyên tắc one-factor-at-a-time: mỗi cấu hình sweep chỉ thay đổi một tham số, hoặc một cặp tham số có liên hệ trực tiếp như `lr` và `lr_decay`, `batch_size` và `buffer_cap`. Tuy nhiên, trong `eval.ipynb`, tên cấu hình khi tổng hợp kết quả không còn được suy ra bằng cách so sánh với baseline. Thay vào đó, notebook tạo tên từ toàn bộ chữ ký tham số thực tế của từng run, giúp tránh mơ hồ khi gộp kết quả từ nhiều máy hoặc nhiều lần chạy.

Các nhóm tham số được khảo sát bao gồm:

| Nhóm | Tham số trong cấu hình | Ý nghĩa |
|---|---|---|
| `gamma` | `agent.gamma` | Hệ số chiết khấu phần thưởng tương lai |
| `learn_every` | `training.learn_every` | Tần suất cập nhật gradient trong quá trình tương tác với môi trường |
| `lr` | `agent.lr`, `agent.lr_decay` | Learning rate và lịch giảm learning rate |
| `batch_size`/`buffer_cap` | `agent.batch_size`, `agent.buffer_cap` | Kích thước batch và dung lượng replay buffer |
| `tau` | `agent.tau` | Tốc độ cập nhật target network |
| `eps_decay` | `agent.eps_decay` | Tốc độ giảm epsilon trong chiến lược exploration |

Cần phân biệt bốn mức cấu hình trong báo cáo này:

1. **Cấu hình baseline:** cấu hình gốc của agent, môi trường giao dịch và quy trình huấn luyện.
2. **Cấu hình sweep:** 20 cấu hình được định nghĩa trước trong `main.ipynb`, mỗi cấu hình thay đổi một phần của baseline.
3. **Run hợp lệ trong đánh giá:** các kết quả thực tế được quét từ `logs.json`, có thể bao gồm các lần chạy lặp lại hoặc kết quả từ nhiều phiên huấn luyện.
4. **Unique config trong `df_merged`:** bảng sau khi khử trùng lặp theo chữ ký cấu hình; nếu nhiều run có cùng chữ ký, `eval.ipynb` giữ run có `WinRate%` cao nhất rồi mới sắp xếp bảng cuối theo `Sharpe`.

### 3.1 Danh sách cấu hình sweep

| Nhóm | Tên cấu hình | Thay đổi so với baseline | Giả thuyết kiểm định |
|---|---|---|---|
| `gamma` | `gamma_090` | `agent.gamma=0.90` | Ưu tiên phần thưởng ngắn hạn hơn, có thể phù hợp với giao dịch ngắn hạn nhưng bỏ qua xu hướng dài hơn |
| `gamma` | `gamma_095` | `agent.gamma=0.95` | Cân bằng hơn giữa tín hiệu ngắn hạn và dài hạn so với 0.90 |
| `gamma` | `gamma_097` | `agent.gamma=0.97` | Gần với baseline, dùng làm mốc so sánh cho discount factor |
| `gamma` | `gamma_099` | `agent.gamma=0.99` | Ưu tiên phần thưởng dài hạn hơn, có thể giúp nắm giữ vị thế theo xu hướng nhưng dễ nhạy với nhiễu |
| `learn_every` | `learn_every_2` | `training.learn_every=2` | Cập nhật gradient thường xuyên hơn, có thể học nhanh hơn nhưng tăng rủi ro overfitting nhiễu ngắn hạn |
| `learn_every` | `learn_every_4` | `training.learn_every=4` | Giá trị baseline, cân bằng giữa tốc độ học và chi phí tính toán |
| `learn_every` | `learn_every_8` | `training.learn_every=8` | Cập nhật ít hơn, có thể ổn định hơn nhưng phản ứng chậm với tín hiệu mới |
| `lr` | `lr_high` | `agent.lr=0.001`, `agent.lr_decay=0.998` | Học nhanh hơn và giảm learning rate chậm hơn, nhưng có nguy cơ dao động Q-value |
| `lr` | `lr_mid` | `agent.lr=0.0005`, `agent.lr_decay=0.995` | Cấu hình learning rate trung bình, trùng với baseline agent |
| `lr` | `lr_low` | `agent.lr=0.0003`, `agent.lr_decay=0.999` | Học chậm và decay rất chậm, có thể ổn định hơn nhưng cần nhiều episode hơn để hội tụ |
| `batch`/`buffer` | `batch128_buf50k` | `agent.batch_size=128`, `agent.buffer_cap=50000` | Batch nhỏ hơn và buffer lớn hơn, tăng đa dạng mẫu nhưng gradient nhiễu hơn |
| `batch`/`buffer` | `batch256_buf20k` | `agent.batch_size=256`, `agent.buffer_cap=20000` | Cấu hình baseline của batch và buffer, dùng làm mốc so sánh |
| `batch`/`buffer` | `batch256_buf50k` | `agent.batch_size=256`, `agent.buffer_cap=50000` | Giữ batch lớn và tăng buffer để kiểm tra tác động của kinh nghiệm lịch sử dài hơn |
| `tau` | `tau_001` | `agent.tau=0.001` | Target network cập nhật rất chậm, giúp mục tiêu Q ổn định nhưng có thể chậm thích nghi |
| `tau` | `tau_005` | `agent.tau=0.005` | Giá trị baseline, cân bằng giữa ổn định và thích nghi |
| `tau` | `tau_01` | `agent.tau=0.01` | Target network cập nhật nhanh hơn, có thể giúp bắt kịp chính sách mới nhưng tăng rủi ro dao động |
| `eps_decay` | `eps_decay_990` | `agent.eps_decay=0.990` | Giảm exploration nhanh, agent chuyển sang exploitation sớm hơn |
| `eps_decay` | `eps_decay_995` | `agent.eps_decay=0.995` | Giảm exploration vừa phải, gần với baseline nhưng khai thác sớm hơn 0.997 |
| `eps_decay` | `eps_decay_997` | `agent.eps_decay=0.997` | Giá trị baseline, cân bằng exploration và exploitation |
| `eps_decay` | `eps_decay_999` | `agent.eps_decay=0.999` | Duy trì exploration lâu hơn, có thể tránh hội tụ sớm vào chính sách kém |

### 3.2 Quan hệ giữa cấu hình sweep và run đánh giá

Mỗi cấu hình sweep trong `main.ipynb` có một tên ngắn để diễn giải mục đích thực nghiệm, ví dụ `lr_high` hoặc `tau_01`. Khi tổng hợp bằng `eval.ipynb`, tên hiển thị trong bảng kết quả được tạo lại từ các giá trị tham số thực tế để phản ánh đầy đủ chữ ký cấu hình. Các khóa được viết tắt như sau:

| Viết tắt | Tham số gốc |
|---|---|
| `g` | `agent.gamma` |
| `lr` | `agent.lr` |
| `lrd` | `agent.lr_decay` |
| `tau` | `agent.tau` |
| `ed` | `agent.eps_decay` |
| `bs` | `agent.batch_size` |
| `buf` | `agent.buffer_cap` |
| `le` | `training.learn_every` |

Ví dụ, một tên cấu hình có thể có dạng `g0.97_lr0.0005_lrd0.995_tau0.01_ed0.997_bs256_buf20k_le4`. Cách đặt tên này dài hơn các nhãn sweep ban đầu nhưng giảm rủi ro nhầm lẫn khi gộp kết quả từ nhiều máy.

Sau khi tạo tên, `eval.ipynb` khử trùng lặp bằng cách giữ run có `WinRate%` cao nhất cho mỗi chữ ký cấu hình. Bảng `df_merged` cuối cùng được sắp xếp theo `Sharpe` để xác định cấu hình đứng đầu.

## 4. Chỉ số đánh giá

Các cấu hình được so sánh theo nhiều chỉ số để tránh đánh giá một chiều:

| Chỉ số | Diễn giải |
|---|---|
| Return % | Lợi nhuận tích lũy trên tập test |
| Sharpe | Lợi nhuận điều chỉnh theo rủi ro; được dùng làm chỉ số xếp hạng chính |
| WinRate % | Tỷ lệ giao dịch có lãi |
| MaxDD % | Maximum drawdown, tức mức sụt giảm vốn lớn nhất |
| N_trades | Số giao dịch trung bình |
| PF | Profit factor, bằng tổng lợi nhuận chia tổng thua lỗ |
| Score | Điểm lựa chọn checkpoint/best model trong pipeline huấn luyện |

Cần diễn giải profit factor một cách thận trọng. Trong bảng kết quả, một số cấu hình có PF rất lớn do tổng thua lỗ rất nhỏ hoặc số giao dịch giới hạn. Vì vậy, PF không nên được sử dụng độc lập để kết luận hiệu quả chiến lược.

## 5. Kết quả tổng quan

Bảng sau trình bày 10 cấu hình đứng đầu theo Sharpe ratio:

| Hạng | Experiment | Return % | Sharpe | WinRate % | MaxDD % | N_trades | PF | Score |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `tau_01_run1` | +6.03 | +0.6008 | 59.7 | -5.40 | 19 | 2.17 | 0.3456 |
| 2 | `baseline_run1` | +2.73 | +0.4937 | 70.2 | -5.01 | 14 | 14329250001.07 | 0.3758 |
| 3 | `learn_every_2` | +1.99 | +0.3942 | 65.3 | -5.30 | 14 | 8126500001.28 | 0.3679 |
| 4 | `eps_decay_999` | +3.13 | +0.2262 | 59.9 | -6.39 | 17 | 6.90 | 0.4455 |
| 5 | `gamma_99` | +2.49 | +0.2082 | 61.2 | -8.27 | 18 | 2.72 | 0.3730 |
| 6 | `baseline_run4` | +1.97 | +0.0123 | 60.2 | -6.84 | 25 | 1.66 | 0.4521 |
| 7 | `batch_size_128_buffer_cap_50000_run1` | +1.14 | -0.0835 | 61.3 | -6.76 | 21 | 1.69 | 0.4632 |
| 8 | `lr_001_lr_decay_998_run2` | +1.42 | -0.1322 | 61.0 | -7.98 | 19 | 2.40 | 0.4086 |
| 9 | `gamma_9` | +1.48 | -0.1797 | 56.1 | -8.26 | 16 | 1.64 | 0.4398 |
| 10 | `baseline_run3` | -0.46 | -0.2098 | 65.9 | -7.90 | 16 | 1.98 | 0.4692 |

Cấu hình `tau_01_run1` có Sharpe cao nhất và lợi nhuận trung bình cao nhất trong nhóm các cấu hình đứng đầu. Đáng chú ý, cấu hình này không có win rate cao nhất; `baseline_run1` đạt win rate 70.2%, cao hơn đáng kể. Tuy nhiên, `tau_01_run1` đạt sự cân bằng tốt hơn giữa lợi nhuận, Sharpe, drawdown và profit factor.

Trong 34 cấu hình, chỉ một số ít đạt Sharpe dương. Nhiều cấu hình có lợi nhuận âm và Sharpe âm, cho thấy hiệu năng của DQN nhạy với lựa chọn siêu tham số và có thể chịu ảnh hưởng lớn từ stochasticity của quá trình huấn luyện.

Phần trực quan hóa tổng quan trong `eval.ipynb` sử dụng heatmap để cho thấy sự phân hóa giữa các cấu hình theo Sharpe, Return, WinRate, MaxDD và PF. Các metric được chuẩn hóa theo từng cột để phục vụ tô màu, trong đó MaxDD được đảo chiều để giá trị trực quan cao hơn tương ứng với mức drawdown tốt hơn. Khi sắp xếp theo Sharpe, cấu hình đứng đầu có profile metric cân bằng hơn so với phần lớn các cấu hình còn lại.

![Tổng quan hiệu năng các cấu hình qua heatmap](../artifacts/outputs/heatmap_overview.png)

*Hình 1. Heatmap tổng quan các cấu hình sau khi khử trùng lặp, sắp xếp theo Sharpe. Màu sắc thể hiện giá trị đã chuẩn hóa theo từng metric, còn chữ trong ô là giá trị gốc.*

## 6. Phân tích theo nhóm siêu tham số

Phần phân tích theo nhóm tham số trong `eval.ipynb` hiện sử dụng biểu đồ cột nhóm (`grouped_bar_by_param.png`) thay cho biểu đồ facet cũ. Biểu đồ này bám trực tiếp theo các nhóm cấu hình được định nghĩa trong `main.ipynb`, gồm `gamma`, `learn_every`, `lr`, `batch_size`, `tau` và `eps_decay`.

Mỗi subplot tương ứng với một nhóm tham số. Trên mỗi subplot, trục hoành là các giá trị tham số được thử nghiệm, còn các cột màu biểu diễn bốn metric:

- `Sharpe`
- `Return%`
- `WinRate%`
- `PF`

Các metric được chuẩn hóa về khoảng `[0, 1]` trên toàn bộ `df_merged`, không chuẩn hóa riêng trong từng nhóm. Vì vậy, biểu đồ cho phép so sánh tương đối giữa các nhóm và giữa các giá trị tham số trên cùng một thang đo. Tuy nhiên, đây là biểu đồ phục vụ nhận diện xu hướng; các kết luận định lượng vẫn cần dựa trên bảng giá trị gốc.

![Phân tích hiệu năng theo từng nhóm siêu tham số](../artifacts/outputs/grouped_bar_by_param.png)

*Hình 2. Biểu đồ cột nhóm so sánh Sharpe, Return%, WinRate% và PF sau chuẩn hóa `[0, 1]` theo từng nhóm tham số được định nghĩa trong `main.ipynb`.*

### 6.1 Nhóm `gamma`

Nhóm `gamma` kiểm tra tác động của discount factor với các giá trị `0.90`, `0.95`, `0.97` và `0.99`. Về mặt diễn giải, `gamma` thấp khiến agent ưu tiên phần thưởng ngắn hạn hơn, trong khi `gamma` cao làm chính sách nhạy hơn với kỳ vọng phần thưởng dài hạn. Biểu đồ grouped bar giúp quan sát đồng thời Sharpe, Return, WinRate và PF để tránh kết luận chỉ dựa trên một metric riêng lẻ.

### 6.2 Nhóm `learn_every`

Nhóm `learn_every` gồm các giá trị `2`, `4` và `8`, phản ánh tần suất cập nhật gradient. Giá trị nhỏ hơn đồng nghĩa với cập nhật thường xuyên hơn, có thể giúp agent thích nghi nhanh hơn nhưng cũng tăng rủi ro học theo nhiễu ngắn hạn. Giá trị lớn hơn làm quá trình cập nhật thưa hơn, có thể ổn định hơn nhưng phản ứng chậm hơn với tín hiệu mới.

### 6.3 Nhóm `lr`

Nhóm `lr` trong sweep kết hợp learning rate và learning rate decay, gồm các cấu hình tương ứng với `0.001`, `0.0005` và `0.0003`. Biểu đồ grouped bar cho phép kiểm tra liệu learning rate cao, trung bình hoặc thấp có tạo ra trade-off giữa Return, Sharpe và WinRate hay không. Do learning rate ảnh hưởng trực tiếp đến độ ổn định của Q-value, nhóm này cần được diễn giải cùng đường cong validation ở phần sau.

### 6.4 Nhóm `batch_size`

Nhóm `batch_size` trong biểu đồ tập trung vào các giá trị `128` và `256`. Trong các cấu hình sweep gốc, batch size thường đi kèm replay buffer capacity, ví dụ `batch128_buf50k`, `batch256_buf20k` và `batch256_buf50k`. Vì biểu đồ nhóm theo từng tham số chính, phần này nên được hiểu là lát cắt theo batch size; ảnh hưởng của `buffer_cap` vẫn có thể lẫn trong cấu hình tương ứng.

### 6.5 Nhóm `tau`

Nhóm `tau` gồm `0.001`, `0.005` và `0.01`, kiểm tra tốc độ soft update của target network. `tau` nhỏ giúp target network ổn định hơn nhưng có thể chậm thích nghi, trong khi `tau` lớn cập nhật nhanh hơn nhưng có nguy cơ làm mục tiêu Q dao động. Việc so sánh đồng thời Sharpe, Return, WinRate và PF giúp đánh giá liệu tốc độ cập nhật nhanh hơn có cải thiện lợi nhuận điều chỉnh theo rủi ro hay chỉ cải thiện một metric đơn lẻ.

### 6.6 Nhóm `eps_decay`

Nhóm `eps_decay` gồm `0.990`, `0.995`, `0.997` và `0.999`, phản ánh tốc độ giảm exploration trong epsilon-greedy. Giá trị gần 1 duy trì exploration lâu hơn, có thể giúp agent tránh hội tụ sớm vào chính sách kém; giá trị thấp hơn khiến agent chuyển sang exploitation nhanh hơn. Biểu đồ grouped bar hỗ trợ đánh giá trade-off giữa khả năng khám phá và hiệu quả giao dịch trên tập test.

## 7. Phân tích top cấu hình qua đường cong huấn luyện

`eval.ipynb` hiện không còn sử dụng radar chart cho top cấu hình. Thay vào đó, notebook tập trung vào đường cong validation của Top 5 cấu hình sau khi `df_merged` được sắp xếp theo Sharpe. Hai đại lượng được vẽ là `val_sharpe` và `val_return`.

Để giảm nhiễu trong quá trình quan sát, mỗi đường cong gồm hai lớp: đường raw được vẽ mờ và đường làm mượt bằng moving average với cửa sổ `10` episode được vẽ đậm. Cách trực quan hóa này giúp đánh giá xu hướng hội tụ, độ dao động và mức độ ổn định của các cấu hình đứng đầu, thay vì chỉ nhìn vào kết quả test cuối cùng.

![Đường cong huấn luyện của các cấu hình đứng đầu](../artifacts/outputs/training_curves_top_n.png)

*Hình 3. Đường cong validation Sharpe và validation Return của Top 5 cấu hình theo Sharpe, sau khi làm mượt bằng moving average với cửa sổ 10 episode.*

## 8. Phân tích cấu hình tốt nhất theo từng mã cổ phiếu

Cấu hình tốt nhất trong tập đánh giá là `tau_01_run1`. Phần dưới trình bày chi tiết hiệu năng của cấu hình này trên từng mã cổ phiếu.

| Mã | Return % | Sharpe | WinRate % | MaxDD % | N_trades | PF | Avg win | Avg loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VNM | -2.03 | -0.7437 | 54.5 | -4.88 | 22 | 1.41 | 2.11 | -1.80 |
| FPT | -6.05 | -1.6900 | 38.9 | -8.39 | 18 | 0.89 | 2.66 | -1.92 |
| VIC | +25.95 | +4.0000 | 78.6 | -4.01 | 14 | 3.72 | 9.49 | -9.35 |
| HPG | +6.24 | +0.8369 | 66.7 | -4.31 | 24 | 2.68 | 2.32 | -1.73 |
| Average | +6.03 | +0.6008 | 59.7 | -5.40 | 19 | 2.17 | - | - |

Kết quả theo từng mã cho thấy hiệu năng trung bình của `tau_01_run1` bị chi phối mạnh bởi VIC. Mã VIC đạt Return +25.95% và Sharpe +4.0000, trong khi HPG cũng đóng góp tích cực với Return +6.24% và Sharpe +0.8369. Ngược lại, VNM và FPT có Return âm và Sharpe âm, đặc biệt FPT có PF 0.89, thấp hơn ngưỡng 1.0.

Điều này cho thấy mô hình chưa đạt tính nhất quán cross-sectional. Nếu mục tiêu là xây dựng chiến lược giao dịch có thể áp dụng rộng trên nhiều cổ phiếu, cần bổ sung các thực nghiệm về tính tổng quát hóa, chẳng hạn train/test theo nhiều tập mã, walk-forward validation và đánh giá trên các giai đoạn thị trường khác nhau.

## 9. Thảo luận

### 9.1 Hiệu quả tổng thể

Kết quả tốt nhất đạt Sharpe dương và lợi nhuận dương trên tập test, cho thấy tác nhân DQN có khả năng học được một số tín hiệu giao dịch có giá trị. So với các cấu hình còn lại, `tau_01_run1` đạt sự cân bằng tốt giữa lợi nhuận và rủi ro, với maximum drawdown ở mức -5.40%.

Tuy nhiên, chỉ một phần nhỏ trong 34 cấu hình đạt Sharpe dương. Điều này cho thấy bài toán có độ khó cao và kết quả phụ thuộc đáng kể vào thiết lập siêu tham số, quá trình huấn luyện và đặc điểm của từng mã cổ phiếu.

### 9.2 Độ nhạy siêu tham số

Kết quả theo nhóm tham số cho thấy `tau`, `learn_every`, `eps_decay` và `gamma` có ảnh hưởng lớn đến hiệu năng. Trong khi đó, các thay đổi về learning rate, replay buffer và batch size trong tập thực nghiệm hiện tại chưa cho thấy cải thiện ổn định. Một điểm đáng chú ý là các run cùng nhóm có thể cho kết quả khác biệt mạnh, vì vậy cần tách ảnh hưởng của siêu tham số khỏi ảnh hưởng của seed và stochasticity.

### 9.3 Rủi ro selection bias

Việc chọn cấu hình tốt nhất trong 34 run có thể tạo ra selection bias: cấu hình đứng đầu có thể là kết quả lạc quan của một lần chạy cụ thể thay vì phản ánh hiệu năng kỳ vọng. Để giảm rủi ro này, mỗi cấu hình nên được lặp lại trên nhiều random seed, sau đó báo cáo trung bình, độ lệch chuẩn và khoảng tin cậy.

### 9.4 Ý nghĩa thực tiễn

Kết quả hiện tại có giá trị như một bằng chứng thực nghiệm ban đầu về tiềm năng của DQN trong bài toán giao dịch cổ phiếu Việt Nam. Tuy nhiên, trước khi ứng dụng thực tế, cần so sánh với các benchmark như buy-and-hold, moving-average crossover, random policy và các mô hình học máy truyền thống. Ngoài ra, cần đánh giá thêm chi phí giao dịch thực tế, thanh khoản, trượt giá động và ràng buộc khớp lệnh.

## 10. Hạn chế

Thực nghiệm hiện tại có một số hạn chế quan trọng:

1. Chưa có benchmark buy-and-hold, random policy hoặc rule-based strategy trong bảng tổng hợp.
2. Số lượng mã cổ phiếu còn nhỏ, chỉ gồm VNM, FPT, VIC và HPG.
3. Chưa báo cáo khoảng thời gian cụ thể của train, validation và test.
4. Một số metric như profit factor có thể bị méo khi tổng thua lỗ rất nhỏ hoặc số giao dịch ít.
5. Chưa có kiểm định thống kê ý nghĩa, ví dụ bootstrap confidence interval hoặc paired test.
6. Các cấu hình chưa được lặp lại đầy đủ trên nhiều seed để ước lượng độ ổn định.
7. Kết quả tốt nhất bị ảnh hưởng mạnh bởi một mã cổ phiếu duy nhất là VIC.

## 11. Hướng nghiên cứu tiếp theo

Để nâng cao độ tin cậy của kết quả, các thực nghiệm tiếp theo nên tập trung vào:

1. Bổ sung benchmark buy-and-hold, random policy và các chiến lược kỹ thuật cơ bản.
2. Lặp lại mỗi cấu hình trên nhiều seed và báo cáo mean, standard deviation, confidence interval.
3. Thực hiện walk-forward validation theo nhiều giai đoạn thị trường.
4. Mở rộng tập cổ phiếu để kiểm tra tính tổng quát hóa cross-sectional.
5. Kiểm định ý nghĩa thống kê của khác biệt giữa các cấu hình.
6. Phân tích ablation cho từng nhóm đặc trưng đầu vào.
7. Tối ưu siêu tham số bằng một quy trình có hệ thống hơn, chẳng hạn Bayesian optimization.

## 12. Kết luận

Kết quả tổng hợp cho thấy cấu hình `tau_01_run1` là cấu hình tốt nhất trong 34 run, đạt Return trung bình +6.03%, Sharpe +0.6008, WinRate 59.7%, MaxDD -5.40% và PF 2.17 trên bốn mã cổ phiếu. Kết quả này cho thấy DQN có tiềm năng học chính sách giao dịch có lời trong một số điều kiện thị trường.

Tuy nhiên, hiệu năng chưa ổn định trên tất cả cổ phiếu và tất cả cấu hình. Do đó, kết quả nên được xem là bằng chứng thực nghiệm ban đầu thay vì kết luận cuối cùng về khả năng vượt trội của mô hình. Để đạt chuẩn công bố khoa học cao hơn, cần bổ sung benchmark, lặp lại trên nhiều seed, kiểm định thống kê và đánh giá out-of-sample nghiêm ngặt hơn.

## Phụ lục A. Nguyên tắc tái lập thực nghiệm

Để tái lập kết quả, cần giữ cố định các yếu tố sau: danh sách mã cổ phiếu, nguồn dữ liệu, tỷ lệ chia train/validation/test, cấu hình môi trường giao dịch, kiến trúc DQN, không gian siêu tham số và quy tắc chọn checkpoint tốt nhất. Khi báo cáo kết quả tái lập, nên công bố thêm random seed, số lần lặp lại mỗi cấu hình, trung bình, độ lệch chuẩn và khoảng tin cậy của các metric chính.
