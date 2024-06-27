(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects k h i l)
(:init 
(handempty)
(ontable k)
(ontable h)
(ontable i)
(ontable l)
(clear k)
(clear h)
(clear i)
(clear l)
)
(:goal
(and
(on k h)
(on h i)
(on i l)
)))