(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects h i k j)
(:init 
(handempty)
(ontable h)
(ontable i)
(ontable k)
(ontable j)
(clear h)
(clear i)
(clear k)
(clear j)
)
(:goal
(and
(on h i)
(on i k)
(on k j)
)))