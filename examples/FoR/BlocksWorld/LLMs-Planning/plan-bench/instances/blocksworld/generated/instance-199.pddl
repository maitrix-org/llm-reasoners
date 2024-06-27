(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects i b g)
(:init 
(handempty)
(ontable i)
(ontable b)
(ontable g)
(clear i)
(clear b)
(clear g)
)
(:goal
(and
(on i b)
(on b g)
)))