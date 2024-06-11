

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(on a d)
(on b a)
(on c e)
(ontable d)
(on e b)
(clear c)
)
(:goal
(and
(on d a)
(on e b))
)
)


