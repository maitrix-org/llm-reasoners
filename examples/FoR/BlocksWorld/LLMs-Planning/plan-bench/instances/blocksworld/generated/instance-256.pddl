(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects k a j)
(:init 
(handempty)
(ontable k)
(ontable a)
(ontable j)
(clear k)
(clear a)
(clear j)
)
(:goal
(and
(on k a)
(on a j)
)))