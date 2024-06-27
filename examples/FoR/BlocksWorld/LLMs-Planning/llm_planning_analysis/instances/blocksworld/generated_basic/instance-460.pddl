

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(on a c)
(ontable b)
(ontable c)
(on d e)
(on e b)
(clear a)
(clear d)
)
(:goal
(and
(on a d)
(on b e)
(on c b))
)
)


