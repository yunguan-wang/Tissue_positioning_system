import inspect
from collections import OrderedDict
from .layers import *

def trace_layer(x, path):
    """ Trace layer in a module from given path. 
        x: nn.Module
        path: list of key or index: ['conv', 1, -1,]
    """
    for _ in path:
        if isinstance(_, str):
            x = getattr(x, _)
        elif isinstance(_, int):
            x = x[_]
    return x

class MobileNetFeatures(nn.Module):
    def __init__(self, architecture='mobilenet_v2', pretrained=False, progress=False, 
                 in_channels=None, **kwargs):
        """ Get features on different scale from Mobilenet (v1, v2) backbone.
            **kwargs parameters:
                width_mult=1.0: adjusts number of channels in each layer by this amount
                setting=None: Network structure. (inverted_residual_setting)
                round_nearest=8: Round the number of channels in each layer to be a multiple of this number.
                block=None: InvertedResidual
                num_classes=1000: num_classes in final FC, not used for Backbone.
        """
        super().__init__()
        if architecture in ['mobilenet_v1', 'mobilenet_v2']:
            self.architecture = self.default_settings(architecture)
        elif architecture is None:
            self.architecture = {}
        else:
            self.architecture = architecture
        
        ## Update architecture with kwargs/defaults
        assert isinstance(self.architecture, dict)
        self.architecture.update(kwargs)
        self.architecture.setdefault('init_block', torchvision.models.mobilenetv2.ConvBNReLU)
        self.architecture.setdefault('width_mult', 1.0)
        self.architecture.setdefault('round_nearest', 8)
        self.architecture.setdefault('return_layers', None)
        assert 'setting' in self.architecture
        assert 'block' in self.architecture
        
        if self.architecture['return_layers'] is None:
            all_layers = np.cumsum([0] + [_[2] for _ in self.architecture['setting']]) + 1
            self.architecture['return_layers'] = all_layers.tolist()
        
        ## build mobilnet classifier
        self.init_block = self.architecture['init_block']
        self.feature_block = self.architecture['block']
        self.backbone = getattr(torchvision.models, 'mobilenet_v2')(
            pretrained, progress, 
            inverted_residual_setting=self.architecture['setting'], 
            block=self.architecture['block'],
            width_mult=self.architecture['width_mult'],
            round_nearest=self.architecture['round_nearest'],
        ).features
        
        ## switch channel for the first conv layer
        if in_channels is not None:
            self.in_channels = in_channels
            if pretrained is True:
                print("pretrained conv1 layer is replaced with random weights. ")
            attrs = ['in_channels', 'out_channels', 'kernel_size', 'stride', 
                     'padding', 'dilation', 'groups', 'bias', 'padding_mode']
            args = dict([(_, getattr(self.backbone[0][0], _)) for _ in attrs])
            args.update({'in_channels': self.in_channels})
            self.backbone[0][0] = nn.Conv2d(**args)
        else:
            self.in_channels = self.backbone[0][0].in_channels
        
        ## remove last ConvBNReLU from feature layers
        feature_layers = list(self.backbone.named_children())[:-1]
        ## Replace the init_block if non-default init_block is provided
        # if init_block is not a class (then should be a function)
        # or init_block is a class, and feature_layers [0][1] is not an instance of this class.
        
        if not (inspect.isclass(self.init_block) and isinstance(feature_layers[0][1], self.init_block)):
            init_name, init_conv = feature_layers[0][0], feature_layers[0][1][0]
            attrs = ['in_channels', 'out_channels', 'kernel_size', 'stride']
            args = dict([(_, getattr(init_conv, _)) for _ in attrs])
            args.update({'in_channels': self.in_channels})
            feature_layers[0] = (init_name, self.init_block(**args))
        self.backbone = nn.Sequential(OrderedDict(feature_layers))
        self.block_channels = [
            trace_layer(feature_layers[0][1], self.architecture['trace']['init_block']).out_channels,
            *[trace_layer(v, self.architecture['trace']['block']).out_channels for k, v in feature_layers[1:]],
        ]
        
        ## Retrive out_channels for all returned feature layers
        self.return_layers = self.architecture['return_layers']
        self.out_channels = [self.block_channels[_-1] for _ in self.return_layers]
#         self.out_channels = [
#             trace_layer(self.backbone, [_-1] + self.architecture['trace']).out_channels 
#             for _ in self.return_layers
#         ]
    
    @staticmethod
    def default_settings(architecture):
        init_block = (lambda in_channels, out_channels, kernel_size=3, stride=1, norm_layer='batch':
                      Conv2dBNReLU(in_channels, out_channels, kernel_size, stride, padding='default', 
                                   norm_layer=norm_layer, activation=nn.ReLU(inplace=True))
                     )
        # init_block = torchvision.models.mobilenetv2.ConvBNReLU
        v2_block = (lambda in_c, out_c, stride=1, kernel_size=3, expand_ratio=1, norm_layer='batch': 
                    InvertedResidual(
                        in_c, out_c, kernel_size, stride, padding='default',
                        expand_ratio=expand_ratio, norm_layer=norm_layer, 
                        activation=nn.ReLU6(inplace=True))
                   )
        # v2_block = torchvision.models.mobilenetv2.InvertedResidual()
        v1_block = DepthwiseConv2dBNReLU6

        settings = {
            'mobilenet_v2': {
                'setting': [
                    # t, c, n, s
                    [1, 16, 1, 1],
                    [6, 24, 2, 2],
                    [6, 32, 3, 2],
                    [6, 64, 4, 2],
                    [6, 96, 3, 1],
                    [6, 160, 3, 2],
                    [6, 320, 1, 1],
                ], 
                'block': v2_block,
                'init_block': init_block, 
                'return_layers': [2, 4, 7, 14, 18],
                'trace': {'init_block': [0], 'block': ['conv', -2]},
            }, 
            'mobilenet_v1': {
                'setting': [
                    # t, c, n, s
                    [1, 64, 1, 1],
                    [1, 128, 2, 2],
                    [1, 256, 2, 2],
                    [1, 512, 6, 2],
                    [1, 1024, 2, 2],
                ],
                'block': v1_block, 
                'init_block': init_block, 
                'return_layers': [2, 4, 6, 12, 14],
                'trace': {'init_block': [0], 'block': [0, -1]},
            }
        }
        
        return settings[architecture]
    
    def feature_channels(self, idx=None):
        if isinstance(idx, int):
            return self.out_channels[idx]
        if idx is None:
            idx = range(len(self.out_channels))
        return [self.out_channels[k] for k in idx]
    
#     def forward(self, x):
#         res = []
#         for s, t in zip([0] + self.return_layers, self.return_layers):
#             for idx in range(s, t):
#                 x = self.backbone[idx](x)
#             res.append(x)
        
#         return res

    def forward(self, x):
        res = []
        for idx, layer in enumerate(self.backbone):
            x = layer(x)
            if idx + 1 in self.return_layers:
                res.append(x)

        return res


class MobileUNet(nn.Module):
    def __init__(self, num_classes, scale_factor=2, resize_mode='bilinear',
                 encoder={'architecture': 'mobilenet_v1', 'width_mult': 1.0, 'return_layers': None}, 
                 decoder={'mode': 'bilinear', 'n_blocks': None}, 
                 out_channels=None, return_logits=False, **kwargs):
        """ Mobile UNet class. 
            num_classes: num_classes in output layer.
            scale_factor: output scale_factor of last layer.
            encoder: nn.Module or dictionary. (call self.get_encoder(**encoder)).
            decoder: nn.Module or dictionary. (call self.get_decoder(**decoder)).
            out_channels: self.encoder.feature_channels() if None
        """
        super().__init__()
        self.num_classes = num_classes
        self.resize_mode = resize_mode
        
        ## encoder
        if isinstance(encoder, dict):
            self.encoder = self.get_encoder(**encoder)
        else:
            self.encoder = encoder
        assert isinstance(self.encoder, nn.Module)
        
        ## out_channels
        self.out_channels = out_channels or self.encoder.out_channels
        
        ## decoder
        if isinstance(decoder, dict):
            self.decoder = self.get_decoder(**decoder)
        else:
            self.decoder = decoder
        assert isinstance(self.decoder, nn.Module)

        ## final classification and resize layer
        classifier = [
            nn.Conv2d(self.out_channels[0], num_classes, kernel_size=1),
        ]
        
        if not return_logits:
            classifier.append((nn.Softmax2d() if num_classes > 1 else nn.Sigmoid()))
        
        ## pytorch interpolate == tf interpolate != keras.Upsample/tf.js.Upsample.
        ## Will see differences on resize in keras and pytorch.
        if scale_factor is not None and scale_factor != 1:
            classifier = [
                nn.Upsample(scale_factor=scale_factor, mode=self.resize_mode)
            ] + classifier
        
        self.classifier = nn.Sequential(*classifier)

    def get_encoder(self, **kwargs):
        ## by default use all possible layers in settings
        kwargs.setdefault('return_layers', None)
        return MobileNetFeatures(**kwargs)
    
    def _default_uplayer(self):
        def ConvTranspose2dUp(in_c, out_c, stride, **kwargs):
            return nn.ConvTranspose2d(in_c, out_c, kernel_size=stride, stride=stride)

        return ConvTranspose2dUp
    
    def get_decoder_config(self, **kwargs):
        config = {
            'setting': kwargs.get('setting') or 'default', 
            'init_block': kwargs.get('init_block', self.encoder.architecture['init_block']),
            'block': kwargs.get('block', self.encoder.architecture['block']),
            'up': kwargs.get('up', self._default_uplayer()),
        }

        if config['setting'] == 'default':
            n_blocks = kwargs.get('n_blocks', None)
            setting = [[1, 32, 1, 2]]
            for t, c, n, s in self.encoder.architecture['setting']:
                setting[-1][-1] = s
                setting.append([t, c, n_blocks or n, s])
            config['setting'] = setting[:-1]
        
        return config
    
    def get_decoder(self, **kwargs):
        N = len(self.out_channels) - 1
        config = self.get_decoder_config(**kwargs)
        assert len(config['setting']) == N, "decoder setting should match encoder layers. "
        
        ## bug in mobilenet v2: setting has s=1 part. So there is a mismatch
        decoder = nn.ModuleList()
        for i in range(N-1, -1, -1):
            in_c = self.out_channels[i+1]
            out_c = self.out_channels[i]
            t, c, n, s = config['setting'][i]
            
            if s == 1 and in_c == out_c:
                up_layer = nn.Identity()
            else:
                up_layer = config['up'](in_c, out_c, stride=s, expand_ratio=t)
            
            block = config['block'] if i > 0 else config['init_block']
            if i > 0:
                block = config['block']
                conv_layer = nn.Sequential(
                    block(out_c*2, out_c, stride=1, expand_ratio=t), 
                    *[block(out_c, out_c, stride=1, expand_ratio=t) for _ in range(n-1)]
                )
            else:
                block = config['init_block']
                conv_layer = nn.Sequential(
                    block(out_c*2, out_c, stride=1), 
                    *[block(out_c, out_c, stride=1) for _ in range(n-1)]
                )

            decoder.append(nn.ModuleDict([('up', up_layer), ('conv', conv_layer)]))

        decoder.architecture = config
        
        return decoder

    def forward(self, x):
        features = self.encoder(x)
        x = features.pop()
        for layers in self.decoder:
            up = layers['up'](x)
            up = torch.cat([up, features.pop()], dim=1)
            x = layers['conv'](up)
        
        return self.classifier(x)
