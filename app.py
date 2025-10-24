import os
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class ImageApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Image Processor")
        self.geometry("600x700")
        self.original_image = None
        self.display_image = None

        # Кнопки
        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(pady=10, padx=20, fill="x")
        ctk.CTkButton(btn_frame, text="Выбрать файл", command=self.load_image).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Преобразовать", command=self.convert_image).pack(side="right", padx=5)

        # Область изображения
        self.image_label = ctk.CTkLabel(
            self,
            text="Нажмите 'Выбрать файл', чтобы загрузить изображение\n(Поддерживаются JPG, PNG)",
            width=400, height=400,
            fg_color=("gray85", "gray25"),
            corner_radius=10,
            justify="center"
        )
        self.image_label.pack(pady=20, padx=20, fill="both", expand=True)

        # Поле вывода
        self.output = ctk.CTkTextbox(self, height=100)
        self.output.pack(pady=10, padx=20, fill="x")
        self.output.insert("0.0", "Результат преобразования появится здесь...")

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png"), ("Все файлы", "*.*")]
        )
        if not path:
            return
        try:
            self.original_image = Image.open(path).convert("RGB")
            self.show_thumbnail()
            self.output.delete("0.0", "end")
            self.output.insert("0.0", f"Загружено: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение:\n{e}")

    def show_thumbnail(self):
        img = self.original_image.copy()
        img.thumbnail((400, 400), Image.LANCZOS)
        self.display_image = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.display_image, text="")

    def convert_image(self):
        if self.original_image is None:
            self.output.delete("0.0", "end")
            self.output.insert("0.0", "⚠️ Сначала загрузите изображение!")
            return

        # Замените это на вашу логику (например, вызов модели)
        result = (
            "Преобразование выполнено!\n"
            f"Размер: {self.original_image.width}×{self.original_image.height}\n"
            "Формат: RGB\n"
            "Дальнейшие действия: можно добавить распознавание формул и т.д."
        )
        self.output.delete("0.0", "end")
        self.output.insert("0.0", result)

if __name__ == "__main__":
    app = ImageApp()
    app.mainloop()