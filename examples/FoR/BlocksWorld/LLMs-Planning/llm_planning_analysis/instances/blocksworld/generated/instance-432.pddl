(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects g b f h)
(:init 
(handempty)
(ontable g)
(ontable b)
(ontable f)
(ontable h)
(clear g)
(clear b)
(clear f)
(clear h)
)
(:goal
(and
(on g b)
(on b f)
(on f h)
)))