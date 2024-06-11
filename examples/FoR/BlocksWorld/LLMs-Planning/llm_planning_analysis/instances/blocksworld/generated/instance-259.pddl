(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects i h f)
(:init 
(handempty)
(ontable i)
(ontable h)
(ontable f)
(clear i)
(clear h)
(clear f)
)
(:goal
(and
(on i h)
(on h f)
)))