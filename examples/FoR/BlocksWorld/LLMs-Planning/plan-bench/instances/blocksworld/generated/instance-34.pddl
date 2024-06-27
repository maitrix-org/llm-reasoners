(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects c d k g)
(:init 
(handempty)
(ontable c)
(ontable d)
(ontable k)
(ontable g)
(clear c)
(clear d)
(clear k)
(clear g)
)
(:goal
(and
(on c d)
(on d k)
(on k g)
)))