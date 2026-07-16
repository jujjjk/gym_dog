/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
  
 
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "fdcan.h"
#include "spi.h"
#include "gpio.h"


/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

#include <string.h>
#include <math.h>
#include <stdbool.h>

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE BEGIN PD */
#ifndef BOARD_IS_A
#define BOARD_IS_A              0
#endif
#define MASTER_CAN_ID           0x00FD

#define SPI_FRAME_LEN           96
#define SPI_CMD_MAGIC           0xFA

#define SPI_OP_NOP              0x00
#define SPI_OP_SEND_CAN_RAW     0x10
#define SPI_OP_REINIT_REPORTS   0x11
#define SPI_OP_SEND_CAN_BATCH   0x12
#define SPI_OP_READ_MOTOR_PARAM 0x13
#define SPI_BATCH_HEADER_LEN    4U
#define SPI_BATCH_ENTRY_LEN     13U
#define SPI_BATCH_MAX_ENTRIES   7U

#define SNAPSHOT_MAGIC          0x5A
#define PARAM_RESPONSE_MAGIC    0x5B
#define SNAPSHOT_CAP_BATCH_CAN  (1UL << 0)
#define SNAPSHOT_CAP_PARAM_READBACK (1UL << 1)
#define ONLINE_TIMEOUT_MS       100U
#define REPORT_RETRY_MS         1000U
#define REPORT_STALE_MS         500U

// EPScan_time: 1 代表 10ms，之后每 +1 增加 5ms
#define ACTIVE_REPORT_PERIOD_CODE   1U

volatile uint32_t g_fdcan_rx_cnt = 0;
volatile uint32_t g_last_ext_id = 0;
volatile uint8_t  g_last_cmd_type = 0;
volatile uint8_t  g_last_src_id = 0;
volatile uint8_t  g_last_dst_id = 0;
volatile uint8_t  g_last_data[8] = {0};
volatile uint32_t g_parse_hit_cnt = 0;
volatile uint32_t g_last_fb_ext_id = 0;
volatile uint8_t  g_last_fb_cmd_type = 0;


volatile uint32_t g_fdcan1_tx_ok = 0, g_fdcan1_tx_fail = 0;
volatile uint32_t g_fdcan2_tx_ok = 0, g_fdcan2_tx_fail = 0;

#if BOARD_IS_A
static const uint8_t BOARD_MOTOR_IDS[6] = {0x11, 0x12, 0x13, 0x21, 0x22, 0x23};
static const uint8_t BOARD_TAG = 0xA1;
#else
static const uint8_t BOARD_MOTOR_IDS[6] = {0x31, 0x32, 0x33, 0x41, 0x42, 0x43};
static const uint8_t BOARD_TAG = 0xB1;
#endif

uint8_t rxbuff[SPI_FRAME_LEN] = {0};
uint8_t txbuff[SPI_FRAME_LEN] = {0};
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */

/* USER CODE END PV */



typedef struct {
    uint8_t  can_id;
    float    pos_rad;
    float    vel_rad_s;
    float    torque_nm;
    float    temp_c;

    uint8_t  online;
    uint8_t  fault_bits;   // bit21~16
    uint8_t  mode_state;   // 0 reset, 1 cali, 2 motor

    uint32_t last_ms;      // 最近一次收到反馈的 ms
    uint32_t rx_count;     // 累计收到反馈次数
} motor_state_t;

typedef struct __attribute__((packed)) {
    uint8_t  can_id;
    uint8_t  online;
    uint8_t  fault_bits;
    uint8_t  mode_state;
    int16_t  pos_mrad;     // rad * 1000
    int16_t  vel_crad;     // rad/s * 100
    int16_t  tq_cNm;       // Nm * 100
    int16_t  temp_dC;      // 摄氏度 * 10
    uint16_t age_ms;       // 当前快照时刻 - last_ms
} motor_snapshot_t;

typedef struct __attribute__((packed)) {
    uint8_t  magic;         // 0x5A
    uint8_t  board_tag;     // A1 / B1
    uint16_t seq;           // 快照序号
    uint32_t board_tick_ms; // 生成快照时刻
    motor_snapshot_t motors[6];
    uint32_t reserved;      // 先保留，凑成 96 字节
} board_snapshot_t;

typedef struct __attribute__((packed)) {
    uint8_t  magic;         // 0x5B
    uint8_t  board_tag;     // A1 / B1
    uint8_t  motor_id;
    uint8_t  status;        // 0 success, 1 waiting, 2 timeout, 3 bad request
    uint16_t index;
    uint16_t reserved0;
    float    value_f32;
    uint32_t response_tick_ms;
    uint8_t  reserved[80];
} param_response_t;

typedef char board_snapshot_size_check[
    (sizeof(board_snapshot_t) == SPI_FRAME_LEN) ? 1 : -1
];
typedef char param_response_size_check[
    (sizeof(param_response_t) == SPI_FRAME_LEN) ? 1 : -1
];

typedef struct {
    volatile uint8_t  pending;
    volatile uint8_t  ready;
    volatile uint8_t  motor_id;
    volatile uint8_t  status;
    volatile uint16_t index;
    volatile float    value_f32;
    volatile uint32_t request_tick_ms;
    volatile uint32_t response_tick_ms;
} param_read_state_t;

static volatile motor_state_t g_motors[6];
static uint16_t g_snapshot_seq = 0;
static volatile param_read_state_t g_param_read = {0};
static uint8_t g_param_response_in_flight = 0;

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MPU_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
static float map_u16_to_float(uint16_t x, float min_v, float max_v)
{
    return ((float)x) * (max_v - min_v) / 65535.0f + min_v;
}

static uint16_t be16_u(const uint8_t *p)
{
    return ((uint16_t)p[0] << 8) | p[1];
}

static int16_t clamp_i16(int32_t x)
{
    if (x > 32767) return 32767;
    if (x < -32768) return -32768;
    return (int16_t)x;
}

static int motor_slot_from_id(uint8_t can_id)
{
    for (int i = 0; i < 6; i++) {
        if (BOARD_MOTOR_IDS[i] == can_id) return i;
    }
    return -1;
}

// 0x1X / 0x3X -> FDCAN1
// 0x2X / 0x4X -> FDCAN2
static uint8_t can_port_from_motor_id(uint8_t can_id)
{
    switch ((can_id >> 4) & 0x0F) {
    case 0x1:
    case 0x3:
        return 1;
    case 0x2:
    case 0x4:
        return 2;
    default:
        return 0;
    }
}

static uint32_t build_ext_id(uint8_t cmd_type, uint16_t data_info_16, uint8_t target_id)
{
    return (((uint32_t)(cmd_type & 0x1F)) << 24) |
           (((uint32_t)data_info_16) << 8) |
           ((uint32_t)target_id);
}

static uint8_t can_send_ext(uint8_t can_port, uint32_t ext_id, const uint8_t data[8])
{
    uint8_t extFrame[4];
    memcpy(extFrame, &ext_id, 4);

    if (can_port == 1) {
        if (FDCAN1_Send_Msg(extFrame, (uint8_t *)data) == 0) { // 假设成功返回 0
            g_fdcan1_tx_ok++;
            return 0;
        } else {
            g_fdcan1_tx_fail++;
            return 1;
        }
    } else if (can_port == 2) {
        if (FDCAN2_Send_Msg(extFrame, (uint8_t *)data) == 0) {
            g_fdcan2_tx_ok++;
            return 0;
        } else {
            g_fdcan2_tx_fail++;
            return 1;
        }
    }
    return 1;
}
static uint8_t motor_write_param_u16(uint8_t can_id, uint16_t index, uint16_t value)
{
    uint8_t port = can_port_from_motor_id(can_id);
    uint32_t ext_id = build_ext_id(0x12, MASTER_CAN_ID, can_id);
    uint8_t data[8] = {0};

    // 类型18：参数写入
    // Byte0~1 index，低字节在前
    // Byte4~7 参数数据，低字节在前
    data[0] = (uint8_t)(index & 0xFF);
    data[1] = (uint8_t)((index >> 8) & 0xFF);

    data[4] = (uint8_t)(value & 0xFF);
    data[5] = (uint8_t)((value >> 8) & 0xFF);
    data[6] = 0;
    data[7] = 0;

    return can_send_ext(port, ext_id, data);
}

static uint8_t motor_read_param(uint8_t can_id, uint16_t index)
{
    uint8_t port = can_port_from_motor_id(can_id);
    uint32_t ext_id = build_ext_id(0x11, MASTER_CAN_ID, can_id);
    uint8_t data[8] = {0};

    data[0] = (uint8_t)(index & 0xFF);
    data[1] = (uint8_t)((index >> 8) & 0xFF);
    return can_send_ext(port, ext_id, data);
}

static uint8_t motor_write_param_u32(uint8_t can_id, uint16_t index, uint32_t value)
{
    uint8_t port = can_port_from_motor_id(can_id);
    uint32_t ext_id = build_ext_id(0x12, MASTER_CAN_ID, can_id);
    uint8_t data[8] = {0};

    data[0] = (uint8_t)(index & 0xFF);
    data[1] = (uint8_t)((index >> 8) & 0xFF);

    data[4] = (uint8_t)(value & 0xFF);
    data[5] = (uint8_t)((value >> 8) & 0xFF);
    data[6] = (uint8_t)((value >> 16) & 0xFF);
    data[7] = (uint8_t)((value >> 24) & 0xFF);

    return can_send_ext(port, ext_id, data);
}

static uint8_t motor_set_active_report(uint8_t can_id, uint8_t enable)
{
    uint8_t port = can_port_from_motor_id(can_id);
    uint32_t ext_id = build_ext_id(0x18, MASTER_CAN_ID, can_id);
    uint8_t data[8] = {1, 2, 3, 4, 5, 6, 0, 0};

    // 手册：F_CMD=0 关闭，F_CMD=1 开启
    data[6] = enable ? 1 : 0;
    data[7] = 0;

    return can_send_ext(port, ext_id, data);
}

static void fdcan_accept_all_rx(void)
{
    if (HAL_FDCAN_ConfigGlobalFilter(
            &hfdcan1,
            FDCAN_ACCEPT_IN_RX_FIFO0,
            FDCAN_ACCEPT_IN_RX_FIFO0,
            FDCAN_REJECT_REMOTE,
            FDCAN_REJECT_REMOTE) != HAL_OK) {
        Error_Handler();
    }

    if (HAL_FDCAN_ConfigGlobalFilter(
            &hfdcan2,
            FDCAN_ACCEPT_IN_RX_FIFO0,
            FDCAN_ACCEPT_IN_RX_FIFO0,
            FDCAN_REJECT_REMOTE,
            FDCAN_REJECT_REMOTE) != HAL_OK) {
        Error_Handler();
    }
}

static void motor_config_reports_one(int i)
{
        uint8_t id = BOARD_MOTOR_IDS[i];

        // 1) 关闭 can timeout，避免没收到控制指令就掉回 reset
        motor_write_param_u32(id, 0x7028, 0);
        HAL_Delay(2);

        // 2) 设置主动上报周期：1 -> 10ms
        motor_write_param_u16(id, 0x7026, ACTIVE_REPORT_PERIOD_CODE);
        HAL_Delay(2);

        // 3) 打开主动上报
        motor_set_active_report(id, 1);
        HAL_Delay(2);
}

static void motor_config_reports_all(void)
{
    for (int i = 0; i < 6; i++) {
        motor_config_reports_one(i);
    }
}

static uint8_t any_motor_feedback_stale(uint32_t now_ms)
{
    for (int i = 0; i < 6; i++) {
        if (g_motors[i].last_ms == 0U) {
            return 1;
        }
        if ((now_ms - g_motors[i].last_ms) > REPORT_STALE_MS) {
            return 1;
        }
    }
    return 0;
}

static void parse_motor_feedback(uint32_t ext_id, const uint8_t data[8])
{
    uint8_t cmd_type = (ext_id >> 24) & 0x1F;
    uint8_t motor_id = (ext_id >> 8)  & 0xFF;
    uint8_t fault    = (ext_id >> 16) & 0x3F;
    uint8_t mode     = (ext_id >> 22) & 0x03;

    // Type 17 (0x11) is a parameter-read response.
    if (cmd_type == 0x11) {
        uint16_t index = ((uint16_t)data[1] << 8) | data[0];
        if (g_param_read.pending &&
            motor_id == g_param_read.motor_id &&
            index == g_param_read.index) {
            float value = 0.0f;
            memcpy(&value, &data[4], sizeof(value));
            g_param_read.value_f32 = value;
            g_param_read.response_tick_ms = HAL_GetTick();
            g_param_read.status = 0;
            g_param_read.pending = 0;
            g_param_read.ready = 1;
        }
        return;
    }

    // 兼容 type 2 和 type 24
    if (!(cmd_type == 0x02 || cmd_type == 0x18)) {
        return;
    }

    int slot = motor_slot_from_id(motor_id);
    if (slot < 0) {
        return;
    }

    g_last_fb_ext_id   = ext_id;
    g_last_fb_cmd_type = cmd_type;
    g_parse_hit_cnt++;

    uint16_t pos_u = be16_u(&data[0]);
    uint16_t vel_u = be16_u(&data[2]);
    uint16_t tq_u  = be16_u(&data[4]);
    uint16_t tmp_u = be16_u(&data[6]);

    g_motors[slot].can_id     = motor_id;
    g_motors[slot].pos_rad    = map_u16_to_float(pos_u, -12.57f, 12.57f);
    g_motors[slot].vel_rad_s  = map_u16_to_float(vel_u, -44.0f, 44.0f);
    g_motors[slot].torque_nm  = map_u16_to_float(tq_u, -17.0f, 17.0f);
    g_motors[slot].temp_c     = ((float)tmp_u) / 10.0f;
    g_motors[slot].fault_bits = fault;
    g_motors[slot].mode_state = mode;
    g_motors[slot].online     = 1;
    g_motors[slot].last_ms    = HAL_GetTick();
    g_motors[slot].rx_count++;
}

static void fill_one_motor_snapshot(motor_snapshot_t *dst, const motor_state_t *src, uint32_t now_ms)
{
    uint32_t age = 0xFFFFu;

    dst->can_id     = src->can_id;
    dst->online     = src->online;
    dst->fault_bits = src->fault_bits;
    dst->mode_state = src->mode_state;

    dst->pos_mrad = clamp_i16((int32_t)lrintf(src->pos_rad * 1000.0f));
    dst->vel_crad = clamp_i16((int32_t)lrintf(src->vel_rad_s * 100.0f));   // 这里改成 *100，避免溢出
    dst->tq_cNm   = clamp_i16((int32_t)lrintf(src->torque_nm * 100.0f));
    dst->temp_dC  = clamp_i16((int32_t)lrintf(src->temp_c * 10.0f));

    if (src->last_ms != 0U) {
        age = now_ms - src->last_ms;
        if (age > 0xFFFFu) age = 0xFFFFu;
    }
    dst->age_ms = (uint16_t)age;
}

static void prepare_snapshot_frame(void)
{
    board_snapshot_t snap;
    motor_state_t local[6];
    uint32_t now_ms = HAL_GetTick();

    __disable_irq();
    for (int i = 0; i < 6; i++) {
        local[i] = g_motors[i];
    }
    __enable_irq();

    memset(&snap, 0, sizeof(snap));
    snap.magic         = SNAPSHOT_MAGIC;
    snap.board_tag     = BOARD_TAG;
    snap.seq           = ++g_snapshot_seq;
    snap.board_tick_ms = now_ms;
    snap.reserved      = SNAPSHOT_CAP_BATCH_CAN |
                         SNAPSHOT_CAP_PARAM_READBACK;

    for (int i = 0; i < 6; i++) {
        fill_one_motor_snapshot(&snap.motors[i], &local[i], now_ms);
    }

    memcpy(txbuff, &snap, sizeof(snap));
}

static void prepare_param_response_frame(void)
{
    param_response_t response;
    memset(&response, 0, sizeof(response));
    response.magic = PARAM_RESPONSE_MAGIC;
    response.board_tag = BOARD_TAG;
    response.motor_id = g_param_read.motor_id;
    response.status = g_param_read.status;
    response.index = g_param_read.index;
    response.value_f32 = g_param_read.value_f32;
    response.response_tick_ms = g_param_read.response_tick_ms;
    memcpy(txbuff, &response, sizeof(response));
    g_param_response_in_flight = 1;
}

static void prepare_spi_tx_frame(void)
{
    if (g_param_read.ready) {
        prepare_param_response_frame();
    } else {
        g_param_response_in_flight = 0;
        prepare_snapshot_frame();
    }
}

static void handle_spi_command(const uint8_t *rx)
{
    if (rx[0] != SPI_CMD_MAGIC) {
        return;
    }

    switch (rx[1]) {
    case SPI_OP_SEND_CAN_RAW:
    {
        // TODO(round3): This legacy opcode forwards exactly one CAN frame per
        // 96-byte SPI frame. Keep it for compatibility, but add a future
        // SPI_OP_SET_TARGETS6 batch opcode so one board can receive six motor
        // targets in a single SPI transaction.
        uint8_t port = rx[2];
        uint32_t ext_id = 0;
        uint8_t data[8];

        memcpy(&ext_id, &rx[4], 4);   // little-endian
        memcpy(data, &rx[8], 8);

        can_send_ext(port, ext_id, data);
        break;
    }

    case SPI_OP_SEND_CAN_BATCH:
    {
        uint8_t count = rx[2];
        if (count > SPI_BATCH_MAX_ENTRIES) {
            count = SPI_BATCH_MAX_ENTRIES;
        }
        for (uint8_t i = 0; i < count; i++) {
            uint32_t offset = SPI_BATCH_HEADER_LEN + ((uint32_t)i * SPI_BATCH_ENTRY_LEN);
            uint8_t port = rx[offset];
            uint32_t ext_id = 0;
            uint8_t data[8];
            memcpy(&ext_id, &rx[offset + 1U], 4);
            memcpy(data, &rx[offset + 5U], 8);
            can_send_ext(port, ext_id, data);
        }
        break;
    }

    case SPI_OP_REINIT_REPORTS:
        motor_config_reports_all();
        break;

    case SPI_OP_READ_MOTOR_PARAM:
    {
        uint8_t motor_id = rx[2];
        uint16_t index = ((uint16_t)rx[5] << 8) | rx[4];
        if (motor_slot_from_id(motor_id) < 0) {
            g_param_read.motor_id = motor_id;
            g_param_read.index = index;
            g_param_read.value_f32 = 0.0f;
            g_param_read.response_tick_ms = HAL_GetTick();
            g_param_read.status = 3;
            g_param_read.pending = 0;
            g_param_read.ready = 1;
            break;
        }
        g_param_read.motor_id = motor_id;
        g_param_read.index = index;
        g_param_read.value_f32 = 0.0f;
        g_param_read.request_tick_ms = HAL_GetTick();
        g_param_read.response_tick_ms = 0;
        g_param_read.status = 1;
        g_param_read.ready = 0;
        g_param_read.pending = 1;
        if (motor_read_param(motor_id, index) != 0) {
            g_param_read.status = 3;
            g_param_read.pending = 0;
            g_param_read.ready = 1;
            g_param_read.response_tick_ms = HAL_GetTick();
        }
        break;
    }

    case SPI_OP_NOP:
    default:
        break;
    }
}
/* USER CODE END 0 */

int main(void)
{
  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MPU Configuration--------------------------------------------------------*/
  MPU_Config();

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_FDCAN2_Init();
  MX_FDCAN1_Init();
  MX_SPI1_Init();
  /* USER CODE BEGIN 2 */
  for (int i = 0; i < 6; i++) {
      g_motors[i].can_id = BOARD_MOTOR_IDS[i];
      g_motors[i].online = 0;
      g_motors[i].fault_bits = 0;
      g_motors[i].mode_state = 0;
      g_motors[i].last_ms = 0;
      g_motors[i].rx_count = 0;
}

// 如果你保留了 fdcan.c 里的 HAL_FDCAN_Start，这里就不要再 Start 了。
// 推荐：只在 main 里 Start 一次。
  fdcan_accept_all_rx();

  HAL_FDCAN_Start(&hfdcan1);
  HAL_FDCAN_Start(&hfdcan2);

  HAL_FDCAN_ActivateNotification(&hfdcan1, FDCAN_IT_RX_FIFO0_NEW_MESSAGE, 0);
  HAL_FDCAN_ActivateNotification(&hfdcan2, FDCAN_IT_RX_FIFO0_NEW_MESSAGE, 0);

  memset(rxbuff, 0, sizeof(rxbuff));
  memset(txbuff, 0, sizeof(txbuff));
  prepare_spi_tx_frame();

 // 给电机一点上电稳定时间，然后打开主动上报
  HAL_Delay(50);
  motor_config_reports_all();
/* USER CODE END 2 */

  /* Infinite loop */
  
  /* USER CODE BEGIN WHILE */
  
  
  
  uint32_t last_report_retry_ms = 0U;
  uint8_t next_report_retry_motor = 0U;

  while (1)
  {
    uint32_t now = HAL_GetTick();

    // 在线超时：超过 ONLINE_TIMEOUT_MS 没刷新就判离线
    for (int i = 0; i < 6; i++) {
        if (g_motors[i].online && (now - g_motors[i].last_ms > ONLINE_TIMEOUT_MS)) {
            g_motors[i].online = 0;
        }
    }

    if (g_param_read.pending &&
        (now - g_param_read.request_tick_ms) > 100U) {
        g_param_read.status = 2;
        g_param_read.pending = 0;
        g_param_read.ready = 1;
        g_param_read.response_tick_ms = now;
    }

    // 先准备当前快照，或一次性参数读回响应。
    prepare_spi_tx_frame();

    // NX 发 96 字节过来时，会在同一帧收到当前 snapshot
    if ((now - last_report_retry_ms) > REPORT_RETRY_MS) {
        last_report_retry_ms = now;
        if (any_motor_feedback_stale(now)) {
            /* Repair one motor per pass. Repairing all six blocks SPI ~36 ms. */
            for (uint8_t step = 0U; step < 6U; step++) {
                uint8_t i = (uint8_t)((next_report_retry_motor + step) % 6U);
                if ((g_motors[i].last_ms == 0U) ||
                    ((now - g_motors[i].last_ms) > REPORT_STALE_MS)) {
                    motor_config_reports_one(i);
                    next_report_retry_motor = (uint8_t)((i + 1U) % 6U);
                    break;
                }
            }
        }
    }

    HAL_SPI_TransmitReceive(&hspi1, txbuff, rxbuff, SPI_FRAME_LEN, 0xFFFF);

    if (g_param_response_in_flight) {
        g_param_read.ready = 0;
        g_param_response_in_flight = 0;
    }

    // 如有命令，则顺手处理
    handle_spi_command(rxbuff);

    memset(rxbuff, 0, sizeof(rxbuff));
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Supply configuration update enable
  */
  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  __HAL_RCC_SYSCFG_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE0);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 5;
  RCC_OscInitStruct.PLL.PLLN = 192;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 8;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_2;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_D3PCLK1|RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV2;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */
void HAL_FDCAN_RxFifo0Callback(FDCAN_HandleTypeDef *hfdcan, uint32_t RxFifo0ITs)
{
    FDCAN_RxHeaderTypeDef rxHeader;
    uint8_t data[8];

    if ((RxFifo0ITs & FDCAN_IT_RX_FIFO0_NEW_MESSAGE) == 0) {
        return;
    }

    while (HAL_FDCAN_GetRxFifoFillLevel(hfdcan, FDCAN_RX_FIFO0) > 0U) {
        if (HAL_FDCAN_GetRxMessage(hfdcan, FDCAN_RX_FIFO0, &rxHeader, data) != HAL_OK) {
            break;
        }

        g_fdcan_rx_cnt++;
        g_last_ext_id   = rxHeader.Identifier;
        g_last_cmd_type = (rxHeader.Identifier >> 24) & 0x1F;
        g_last_src_id   = (rxHeader.Identifier >> 8) & 0xFF;
        g_last_dst_id   = rxHeader.Identifier & 0xFF;
        memcpy((void *)g_last_data, data, 8);

        if (rxHeader.IdType == FDCAN_EXTENDED_ID) {
            parse_motor_feedback(rxHeader.Identifier, data);
        }
    }
}
/* USER CODE END 4 */

/* MPU Configuration */

void MPU_Config(void)
{
  MPU_Region_InitTypeDef MPU_InitStruct = {0};

  /* Disables the MPU */
  HAL_MPU_Disable();

  /** Initializes and configures the Region and the memory to be protected
  */
  MPU_InitStruct.Enable = MPU_REGION_ENABLE;
  MPU_InitStruct.Number = MPU_REGION_NUMBER0;
  MPU_InitStruct.BaseAddress = 0x0;
  MPU_InitStruct.Size = MPU_REGION_SIZE_4GB;
  MPU_InitStruct.SubRegionDisable = 0x87;
  MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL0;
  MPU_InitStruct.AccessPermission = MPU_REGION_NO_ACCESS;
  MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
  MPU_InitStruct.IsShareable = MPU_ACCESS_SHAREABLE;
  MPU_InitStruct.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
  MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;

  HAL_MPU_ConfigRegion(&MPU_InitStruct);
  /* Enables the MPU */
  HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);

}

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
