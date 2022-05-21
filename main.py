from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.graphics import Line
from kivy.properties import ListProperty, NumericProperty
from kivy.graphics import Rectangle, Color, Line
from kivy.properties import StringProperty
import torch
import neural_network_trainer as trainer
import kivy
 

class DrawingWidget(Widget):
    """
     Widget with canvas drawning
    """

    target_colour_rgb = ListProperty([0, 0, 0])
    target_width_px = NumericProperty(0)

    def on_touch_down(self, touch):
        super(DrawingWidget, self).on_touch_down(touch)

        if not self.collide_point(*touch.pos):
            return

        with self.canvas:
            Color(*self.target_colour_rgb)
            self.line = Line(points = [touch.pos[0], touch.pos[1]], width = 20)

    def on_touch_move(self, touch):
        if not self.collide_point(*touch.pos):
            return

        self.line.points = self.line.points + [touch.pos[0], touch.pos[1]]


class MainWidget(BoxLayout):
    """
    Rest of the widget with buttons and label
    """

    predicted_number = StringProperty("Number:")

    def on_click_guess(self,widget: Widget):
        widget.export_to_png("file.png")
        self.predicted_number = "Number: " + str(trainer.guess_number("file.png", torch.load("model.pth")))

    def on_click_clear(self,widget: Widget):
        with widget.canvas:
            Color(1, 1, 1, 1)
            Rectangle(pos = widget.pos, size = widget.size)


class NumberGuesserApp(App):
    """
    App class
    """
    pass


if __name__ == "__main__":
    NumberGuesserApp().run()
    print(kivy.__version__)
