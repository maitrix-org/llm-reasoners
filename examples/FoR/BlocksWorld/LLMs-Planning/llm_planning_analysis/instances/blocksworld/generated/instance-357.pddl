(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects b f i g)
(:init 
(handempty)
(ontable b)
(ontable f)
(ontable i)
(ontable g)
(clear b)
(clear f)
(clear i)
(clear g)
)
(:goal
(and
(on b f)
(on f i)
(on i g)
)))