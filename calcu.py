from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.graphics import Color, Ellipse, Rectangle
from kivy.core.window import Window
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
import socket
import threading
import json
import os


class CircleButton(Button):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ''
        self.background_down = ''
        self.background_color = (0, 0, 0, 0)
        with self.canvas.before:
            self.color_circle = Color(0.9, 0.9, 0.9, 1)
            self.circle = Ellipse(pos=self.pos, size=self.size)
        self.bind(pos=self._update_circle, size=self._update_circle)

    def _update_circle(self, *args):
        self.circle.pos = self.pos
        self.circle.size = self.size

    def set_circle_color(self, rgba):
        self.color_circle.rgba = rgba


class CalculatorApp(App):
    def build(self):
        self.title = 'Máy Tính'
        self.history = []
        Window.size = (400, 700)
        
        # Đường dẫn file lưu cấu hình
        self.config_file = 'printer_config.json'
        
        # Load cấu hình đã lưu hoặc dùng mặc định
        self.load_printer_config()

        main_layout = BoxLayout(orientation='vertical', padding=[10, 10, 10, 10], spacing=10)
        with main_layout.canvas.before:
            Color(1, 1, 1, 1)
            self.bg = Rectangle(size=main_layout.size, pos=main_layout.pos)
        main_layout.bind(size=self._update_bg, pos=self._update_bg)

        # --- Lịch sử ---
        history_scroll = ScrollView(size_hint_y=0.25, do_scroll_x=False, bar_width=10)
        self.history_label = Label(
            text='Lịch sử tính toán:',
            size_hint_y=None,
            height=1500,
            halign='left',
            valign='top',
            font_size='18sp',
            color=(0.2, 0.2, 0.2, 1),
            markup=True,
        )
        self.history_label.bind(size=lambda s, _: setattr(s, 'text_size', (s.width, None)))
        self.history_label.bind(texture_size=lambda i, v: setattr(i, 'height', v[1]))
        history_scroll.add_widget(self.history_label)

        # --- Biểu thức & kết quả ---
        self.expression_label = Label(
            text='',
            size_hint_y=0.07,
            halign='right',
            valign='bottom',
            font_size='20sp',
            color=(0.5, 0.5, 0.5, 1)
        )
        self.expression_label.bind(size=self.expression_label.setter('text_size'))

        self.display = Label(
            text='0',
            size_hint_y=0.11,
            halign='right',
            valign='middle',
            font_size='48sp',
            bold=True,
            color=(0, 0, 0, 1)
        )
        self.display.bind(size=self.display.setter('text_size'))

        # --- Layout nút ---
        buttons_layout = GridLayout(
            cols=4,
            spacing=[10, 10],
            padding=[5, 5, 5, 5],
            size_hint_y=0.57
        )

        buttons = [
            ['7', '8', '9', '/'],
            ['4', '5', '6', '*'],
            ['1', '2', '3', '-'],
            ['AC', '0', '=', '+'],
            ['🖨', '.', '⌫', '🗑']
        ]

        for row in buttons:
            for button_text in row:
                if button_text == '':
                    buttons_layout.add_widget(Label())
                    continue

                btn = CircleButton(
                    text=button_text,
                    font_size='28sp' if button_text not in ['🖨', '⌫', '🗑'] else '32sp',
                    bold=True,
                    color=(0, 0, 0, 1),
                    size_hint=(1, 1)
                )

                # Màu sắc đặc biệt
                if button_text == '🖨':
                    btn.set_circle_color((0.2, 0.6, 1, 1))
                elif button_text == 'AC':
                    btn.set_circle_color((1, 0.3, 0.3, 1))
                elif button_text == '=':
                    btn.set_circle_color((0.2, 0.8, 0.2, 1))
                elif button_text == '⌫':
                    btn.set_circle_color((1, 0.6, 0.2, 1))
                elif button_text == '🗑':
                    btn.set_circle_color((0.9, 0.4, 0.7, 1))
                else:
                    btn.set_circle_color((0.85, 0.85, 0.85, 1))

                btn.bind(on_press=self.on_button_press)
                buttons_layout.add_widget(btn)

        # --- Thêm vào giao diện ---
        main_layout.add_widget(history_scroll)
        main_layout.add_widget(self.expression_label)
        main_layout.add_widget(self.display)
        main_layout.add_widget(buttons_layout)
        return main_layout

    def load_printer_config(self):
        """Load cấu hình máy in từ file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.printer_ip = config.get('ip', '192.168.1.100')
                    self.printer_port = config.get('port', 9100)
            else:
                # Giá trị mặc định
                self.printer_ip = '192.168.1.100'
                self.printer_port = 9100
        except:
            self.printer_ip = '192.168.1.100'
            self.printer_port = 9100

    def save_printer_config(self):
        """Lưu cấu hình máy in vào file"""
        try:
            config = {
                'ip': self.printer_ip,
                'port': self.printer_port
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f)
        except Exception as e:
            print(f"Không thể lưu cấu hình: {e}")

    def _update_bg(self, instance, value):
        self.bg.pos = instance.pos
        self.bg.size = instance.size

    def on_button_press(self, instance):
        button_text = instance.text
        current_display = self.display.text

        if button_text == 'AC':
            self.display.text = '0'
            self.expression_label.text = ''
            self.history = []
            self.update_history_display()

        elif button_text == '=':
            try:
                result = eval(current_display)
                calculation = f"{current_display} = {result}"
                self.history.append(calculation)
                self.update_history_display()
                self.expression_label.text = current_display + ' ='
                self.display.text = str(result)
            except:
                self.display.text = 'Lỗi'
                self.expression_label.text = ''

        elif button_text == '🖨':
            self.print_history()

        elif button_text == '⌫':
            if current_display != '0' and current_display != 'Lỗi':
                if len(current_display) > 1:
                    self.display.text = current_display[:-1]
                    self.expression_label.text = self.display.text
                else:
                    self.display.text = '0'
                    self.expression_label.text = ''

        elif button_text == '🗑':
            if self.history:
                self.history.pop()
                self.update_history_display()

        elif button_text == '.':
            if '.' not in current_display.split()[-1]:
                if current_display == '0':
                    self.display.text = '0.'
                else:
                    self.display.text += '.'
                self.expression_label.text = self.display.text

        else:
            if current_display == '0' or current_display == 'Lỗi':
                self.display.text = button_text
            else:
                self.display.text += button_text
            self.expression_label.text = self.display.text

    def update_history_display(self):
        if self.history:
            lines = ['[b]Lịch sử tính toán:[/b]']
            for i, calc in enumerate(self.history, 1):
                lines.append(f"  {i}. {calc}")
            self.history_label.text = '\n'.join(lines)
        else:
            self.history_label.text = '[b]Lịch sử tính toán:[/b]'

    def show_printer_config_on_error(self, error_msg):
        """Hiển thị popup cấu hình khi có lỗi kết nối"""
        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Thông báo lỗi
        error_label = Label(
            text=f'[color=ff0000]Lỗi kết nối:[/color]\n{error_msg}\n\nVui lòng kiểm tra lại:',
            size_hint_y=0.3,
            color=(1, 0, 0, 1),
            markup=True
        )
        content.add_widget(error_label)
        
        content.add_widget(Label(
            text='Địa chỉ IP máy in:',
            size_hint_y=0.2,
            color=(0, 0, 0, 1)
        ))
        
        ip_input = TextInput(
            text=self.printer_ip,
            multiline=False,
            size_hint_y=0.25,
            font_size='20sp'
        )
        content.add_widget(ip_input)
        
        content.add_widget(Label(
            text='Cổng (Port):',
            size_hint_y=0.2,
            color=(0, 0, 0, 1)
        ))
        
        port_input = TextInput(
            text=str(self.printer_port),
            multiline=False,
            size_hint_y=0.25,
            font_size='20sp'
        )
        content.add_widget(port_input)

        btn_layout = BoxLayout(size_hint_y=0.3, spacing=10)
        
        popup = Popup(
            title='Cấu hình lại máy in',
            content=content,
            size_hint=(0.9, 0.7)
        )
        
        def on_retry(instance):
            self.printer_ip = ip_input.text
            try:
                self.printer_port = int(port_input.text)
            except:
                self.show_message("Lỗi", "Port phải là số!")
                return
            self.save_printer_config()
            popup.dismiss()
            self.print_history()
        
        def on_cancel(instance):
            popup.dismiss()
        
        btn_retry = Button(text='Thử lại', background_color=(0.2, 0.8, 0.2, 1))
        btn_retry.bind(on_press=on_retry)
        
        btn_cancel = Button(text='Hủy', background_color=(1, 0.3, 0.3, 1))
        btn_cancel.bind(on_press=on_cancel)
        
        btn_layout.add_widget(btn_cancel)
        btn_layout.add_widget(btn_retry)
        content.add_widget(btn_layout)
        
        popup.open()

    def show_printer_config(self):
        """Hiển thị popup để cấu hình IP máy in"""
        if not self.history:
            self.show_message("Thông báo", "Không có lịch sử để in!")
            return

        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        content.add_widget(Label(
            text='Địa chỉ IP máy in:',
            size_hint_y=0.3,
            color=(0, 0, 0, 1)
        ))
        
        ip_input = TextInput(
            text=self.printer_ip,
            multiline=False,
            size_hint_y=0.3,
            font_size='20sp'
        )
        content.add_widget(ip_input)
        
        content.add_widget(Label(
            text='Cổng (Port):',
            size_hint_y=0.3,
            color=(0, 0, 0, 1)
        ))
        
        port_input = TextInput(
            text=str(self.printer_port),
            multiline=False,
            size_hint_y=0.3,
            font_size='20sp'
        )
        content.add_widget(port_input)

        btn_layout = BoxLayout(size_hint_y=0.4, spacing=10)
        
        popup = Popup(
            title='Cấu hình máy in',
            content=content,
            size_hint=(0.9, 0.5)
        )
        
        def on_print(instance):
            self.printer_ip = ip_input.text
            self.printer_port = int(port_input.text)
            self.save_printer_config()  # Lưu cấu hình
            popup.dismiss()
            self.print_history()
        
        def on_cancel(instance):
            popup.dismiss()
        
        btn_print = Button(text='In', background_color=(0.2, 0.8, 0.2, 1))
        btn_print.bind(on_press=on_print)
        
        btn_cancel = Button(text='Hủy', background_color=(1, 0.3, 0.3, 1))
        btn_cancel.bind(on_press=on_cancel)
        
        btn_layout.add_widget(btn_cancel)
        btn_layout.add_widget(btn_print)
        content.add_widget(btn_layout)
        
        popup.open()

    def print_history(self):
        """Gửi lịch sử qua mạng LAN đến máy in"""
        if not self.history:
            self.show_message("Thông báo", "Không có lịch sử để in!")
            return

        # Chạy trong thread riêng để không block UI
        thread = threading.Thread(target=self._send_to_printer)
        thread.daemon = True
        thread.start()

    def _send_to_printer(self):
        """Gửi dữ liệu qua socket TCP"""
        try:
            # Tạo nội dung in
            content = self._format_print_content()
            
            # Kết nối đến máy in
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)  # Timeout 3 giây
            sock.connect((self.printer_ip, self.printer_port))
            
            # Gửi dữ liệu
            sock.sendall(content.encode('utf-8'))
            sock.close()
            
            self.show_message("Thành công", f"Đã gửi {len(self.history)} phép tính đến máy in!")
            
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            # Kết nối thất bại → Hiện popup để nhập lại
            from kivy.clock import Clock
            Clock.schedule_once(lambda dt: self.show_printer_config_on_error(str(e)), 0)
        except Exception as e:
            from kivy.clock import Clock
            Clock.schedule_once(lambda dt: self.show_printer_config_on_error(str(e)), 0)

    def _format_print_content(self):
        """Định dạng nội dung để in (ESC/POS hoặc text thuần)"""
        lines = []
        lines.append("=" * 40)
        lines.append("LỊCH SỬ TÍNH TOÁN")
        lines.append("=" * 40)
        
        for i, calc in enumerate(self.history, 1):
            lines.append(f"{i}. {calc}")
        
        lines.append("=" * 40)
        lines.append(f"Tổng: {len(self.history)} phép tính")
        lines.append("=" * 40)
        lines.append("\n\n\n")  # Dòng trống để cắt giấy
        
        # Thêm lệnh cắt giấy cho máy in ESC/POS (nếu hỗ trợ)
        # ESC + 'm' = cắt giấy
        content = "\n".join(lines)
        content += "\x1B\x6D"  # Lệnh cắt giấy ESC/POS
        
        return content

    def show_message(self, title, message):
        """Hiển thị thông báo popup"""
        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        content.add_widget(Label(
            text=message,
            color=(0, 0, 0, 1),
            size_hint_y=0.7
        ))
        
        popup = Popup(
            title=title,
            content=content,
            size_hint=(0.8, 0.3)
        )
        
        btn = Button(
            text='Đóng',
            size_hint_y=0.3,
            background_color=(0.2, 0.6, 1, 1)
        )
        btn.bind(on_press=popup.dismiss)
        content.add_widget(btn)
        
        popup.open()


if __name__ == '__main__':
    CalculatorApp().run()