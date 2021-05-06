import runway
from runway.data_types import number, text, image
from example_model import ExampleModel

# Setup the model, initialize weights, set the configs of the model, etc.
# Every model will have a different set of configurations and requirements.
# Check https://sdk.runwayml.com/en/latest/runway_module.html to see a complete
# list of supported configs. The setup function should return the model ready to
# be used.
setup_options = {
    'truncation': number(min=5, max=100, step=1, default=10),
    'seed': number(min=0, max=1000000)
}
@runway.setup(options=setup_options)
def setup(opts):
    msg = '[SETUP] Run with options: seed = {}, truncation = {}'
    print(msg.format(opts['seed'], opts['truncation']))
    model = ExampleModel(opts)
    return model

# Every model needs to have at least one command. Every command allows to send
# inputs and process outputs. To see a complete list of supported inputs and
# outputs data types: https://sdk.runwayml.com/en/latest/data_types.html
@runway.command(name='generate',
                inputs={ 'caption': text() },
                outputs={ 'image': image(width=512, height=512) })
def generate(model, args):
    print('[GENERATE] Ran with caption value "{}"'.format(args['caption']))
    # Generate a PIL or Numpy image based on the input caption, and return it
    output_image = model.run_on_input(args['caption'])
    return {
        'image': output_image
    }

if __name__ == '__main__':
    # run the model server using the default network interface and ports,
    # displayed here for convenience
    runway.run(host='0.0.0.0', port=8000)
