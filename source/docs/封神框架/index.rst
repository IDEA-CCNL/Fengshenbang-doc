封神框架
=================================


.. toctree::
    :maxdepth: 1

    参数管理


为了让大家好用封神榜大模型，参与大模型的继续训练和下游应用，我们同步开源了FengShen(封神)框架。我们参考了HuggingFace, Megatron-LM, Pytorch-Lightning, DeepSpeed等优秀的开源框架，结合NLP领域的特点, 以Pytorch为基础框架，Pytorch-Lightning为Pipeline重新设计了FengShen。 FengShen可以应用在基于海量数据(TB级别数据)的大模型(百亿级别参数)预训练以及各种下游任务的微调，用户可以通过配置的方式很方便地进行分布式训练和节省显存的技术，更加聚焦在模型实现和创新。同时FengShen也能直接使用HuggingFace中的模型结构进行继续训练，方便用户进行领域模型迁移。FengShen针对封神榜开源的模型和模型的应用，提供丰富、真实的源代码和示例。随着封神榜模型的训练和应用，我们也会不断优化FengShen框架，敬请期待。
