        for i in range(generator.max_iter):
            train_inds, full_train_inds, test_inds = generator.next_task()
            train_sub_loaders_wo_aug.append(
                get_custom_loader(
                    generator.train_dataset_wo_augment, train_inds, batch_size=len(train_inds), shuffle=False)
            )
            test_loaders.append(get_custom_loader(generator.test_dataset, test_inds, batch_size=opts.batch_size))
            if opts.runner_type == 'coreset':
                train_loaders.append(
                    get_custom_loader(generator.train_dataset, train_inds, batch_size=opts.batch_size)
                )
            else:
                raise ValueError('Invalid runner type')
